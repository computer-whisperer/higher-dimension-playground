struct RenderMetadata {
    window_width: u32,
    window_height: u32,
    render_width: u32,
    render_height: u32,
    depth_factor: u32
}

struct Tetrahedron {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    v3: vec4<f32>,
    v0_tex: vec3<f32>,
    v1_tex: vec3<f32>,
    v2_tex: vec3<f32>,
    v3_tex: vec3<f32>,
    texture_id: u32
}

struct TetrahedronMetadata {
    tet_count: u32,
    tets_processed: atomic<u32>
}

const MAX_BUCKET_SIZE: u32 = 512u;
struct OutputBucket {
    tetrahedrons: array<Tetrahedron, MAX_BUCKET_SIZE>,
    num_filled: u32,
}

const NULL_TEXTURE_ID: u32 = 0u;

@group(0) @binding(3) var<uniform> render_metadata : RenderMetadata;

@group(2) @binding(4) var<storage, read_write> incoming_tet_buffer : array<Tetrahedron>;
@group(2) @binding(5) var<storage, read_write> output_tet_buckets : array<OutputBucket>;
@group(2) @binding(7) var<storage, read_write> tet_metadata : TetrahedronMetadata;

const WORKGROUP_SIZE: u32 = 256u;
const READ_BATCH_SIZE: u32 = 1024;
const OUTPUT_CHANNELS: u32 = 256u;

var<workgroup> batch_head: u32; // Automatically initializes to 0
var<workgroup> batch_end: u32; // Automatically initializes to 0

const PRESENCE_TABLE_SIZE = WORKGROUP_SIZE * OUTPUT_CHANNELS/32;
var<workgroup> presence_table: array<u32, PRESENCE_TABLE_SIZE>;

const BUCKETS_X: u32 = 8;
const BUCKETS_Y: u32 = 8;
const BUCKETS_Z: u32 = OUTPUT_CHANNELS/(BUCKETS_X*BUCKETS_Y);

fn clip_to_bucket(pos: vec3<f32>) -> vec3<u32> {
    return vec3<u32>(
        u32(pos.x)/(render_metadata.render_width*BUCKETS_X),
        u32(pos.y)/(render_metadata.render_height*BUCKETS_Y),
        u32(pos.z)/(render_metadata.depth_factor*BUCKETS_Z)
    );
}

fn bucket_to_output_queue_idx(pos: vec3<u32>, dispatch_id: u32, group_count: u32) -> u32 {
    return (((dispatch_id)*group_count + pos.x)*BUCKETS_X + pos.y)*BUCKETS_Y + pos.z;
}

fn bucket_to_presence_idx(bucket: vec3<u32>, idx: u32) -> u32 {
    return ((((idx*BUCKETS_Y + bucket.y)*BUCKETS_X + bucket.x)*BUCKETS_Z + bucket.z));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn raster_sort_main(@builtin(workgroup_id) workgroup_ida: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(num_workgroups) group_counts: vec3<u32>) {
    let instance_id = local_id.x;
    let group_count = group_counts.x;
    let workgroup_id = workgroup_ida.x;
    let lead = instance_id == 0;

    loop {
        // Input phase
        workgroupBarrier();
        if (lead) {
            batch_head += READ_BATCH_SIZE;
            if (batch_head > batch_end)
            {
                batch_head = atomicAdd(&tet_metadata.tets_processed, READ_BATCH_SIZE);
                batch_end = batch_head + READ_BATCH_SIZE;
                batch_end = min(batch_end, tet_metadata.tet_count);
            }
        }
        workgroupBarrier();
        if (batch_head >= batch_end) {
            return;
        }
        // Clear tet buffer
        for (var i = 0u; i < PRESENCE_TABLE_SIZE/WORKGROUP_SIZE; i++) {
            presence_table[i*PRESENCE_TABLE_SIZE/WORKGROUP_SIZE + instance_id] = 0u;
        }

        // Rasterize phase
        workgroupBarrier();
        {
            let tet_id = instance_id;
            let tet = incoming_tet_buffer[batch_head + tet_id];
            if (tet.texture_id != NULL_TEXTURE_ID) {
                let v0_b = clip_to_bucket(tet.v0.xyz);
                let v1_b = clip_to_bucket(tet.v1.xyz);
                let v2_b = clip_to_bucket(tet.v2.xyz);
                let v3_b = clip_to_bucket(tet.v3.xyz);

                let vmax = max(max(max(v0_b, v1_b), v2_b), v3_b);
                let vmin = min(min(min(v0_b, v1_b), v2_b), v3_b);

                for (var x = vmin.x; x < vmax.x; x++) {
                    for (var y = vmin.y; y < vmax.y; y++) {
                        for (var z = vmin.z; z < vmax.z; z++) {
                            let bucket = vec3<u32>(x, y, z);
                            // TODO: Clip to tetrahedron's actual bound (don't pass empty corners)

                            let full_idx = bucket_to_presence_idx(bucket, instance_id);
                            presence_table[full_idx/32u] |= 1u<<full_idx%32u;
                        }
                    }
                }
            }
        }



        // Output phase
        // TODO: Optimize for sparse population
        workgroupBarrier();
        {
            let bucket = vec3<u32>(instance_id % BUCKETS_X, (instance_id / BUCKETS_X) % BUCKETS_Y, instance_id / (BUCKETS_X * BUCKETS_Y) );
            let output_queue_idx = bucket_to_output_queue_idx(bucket, workgroup_id, group_count);
            var num_filled = 0u;

            for (var tet_id = 0u; tet_id < WORKGROUP_SIZE; tet_id++) {
                let full_idx = bucket_to_presence_idx(bucket, tet_id);
                if ((presence_table[full_idx/32u] & 1u<<full_idx%32u) != 0) {
                    let tet = incoming_tet_buffer[batch_head + tet_id];
                    output_tet_buckets[output_queue_idx].tetrahedrons[num_filled] = tet;
                    num_filled++;
                }
            }
            output_tet_buckets[output_queue_idx].num_filled = num_filled;
        }
    }
}