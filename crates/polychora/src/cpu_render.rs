use common::{MatN, ModelInstance, ModelTetrahedron, VecN};
use std::f32::consts::PI;

// ─── Parameters ─────────────────────────────────────────────────────

pub struct CpuRenderParams {
    pub view_matrix: MatN<5>,
    pub focal_length_xy: f32,
    pub focal_length_zw: f32,
    pub width: u32,
    pub height: u32,
    pub sun_dir: [f32; 4],
}

impl Default for CpuRenderParams {
    fn default() -> Self {
        let s = [0.3_f32, 1.0, -0.3, 0.0];
        let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3]).sqrt();
        Self {
            view_matrix: MatN::<5>::identity(),
            focal_length_xy: 1.0,
            focal_length_zw: 1.0,
            width: 120,
            height: 68,
            sun_dir: [s[0] / len, s[1] / len, s[2] / len, s[3] / len],
        }
    }
}

// ─── Internal types ─────────────────────────────────────────────────

#[derive(Clone)]
struct ClipVert {
    pos: [f32; 5], // [x, y, z, w, projDiv]
    tex: [f32; 4],
}

#[derive(Clone)]
struct ClipTet {
    verts: [ClipVert; 4],
    material_id: u32,
}

struct ProjectedTet {
    screen_xy: [[f32; 2]; 4], // NDC x,y per vertex
    zw: [[f32; 2]; 4],        // view-space z,w per vertex (projected[i].zw)
    tex: [[f32; 4]; 4],
    inv_proj_div: [f32; 4],
    normal: [f32; 4],
    material_id: u32,
}

struct ZWLine {
    zw: [[f32; 2]; 2],
    tex: [[f32; 4]; 2],
    normal: [f32; 4],
    material_id: u32,
}

const MIN_DEPTH_DIVISOR: f32 = 0.15;
const DEPTH_FACTOR: usize = 256;

// Face combinations for iterating over tetrahedron faces (4 choose 3)
const COMBOS: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

// ─── Public API ─────────────────────────────────────────────────────

pub fn cpu_render(
    instances: &[ModelInstance],
    model_tets: &[ModelTetrahedron],
    params: &CpuRenderParams,
    content_registry: &polychora::content_registry::ContentRegistry,
) -> image::RgbaImage {
    let projected = clip_and_project(instances, model_tets, params);
    eprintln!(
        "CPU render: {} instances × {} model tets → {} clipped tets",
        instances.len(),
        model_tets.len(),
        projected.len()
    );

    let w = params.width as usize;
    let h = params.height as usize;
    let mut img = image::RgbaImage::new(params.width, params.height);

    // Sun direction in view space
    let sun5 = VecN::<5>::new([
        params.sun_dir[0],
        params.sun_dir[1],
        params.sun_dir[2],
        params.sun_dir[3],
        0.0,
    ]);
    let sv = params.view_matrix * sun5;
    let sun_view = [sv[0], sv[1], sv[2], sv[3]];
    let sun_len = (sun_view[0] * sun_view[0]
        + sun_view[1] * sun_view[1]
        + sun_view[2] * sun_view[2]
        + sun_view[3] * sun_view[3])
        .sqrt();
    let sun_view_dir = if sun_len > 1e-6 {
        [
            sun_view[0] / sun_len,
            sun_view[1] / sun_len,
            sun_view[2] / sun_len,
            sun_view[3] / sun_len,
        ]
    } else {
        [0.0, 1.0, 0.0, 0.0]
    };

    // ZW angle range (same as GPU)
    let zw_view_angle = (PI / 2.0) / params.focal_length_zw;
    let theta_min = PI / 4.0 - zw_view_angle / 2.0;
    let theta_max = PI / 4.0 + zw_view_angle / 2.0;

    for py in 0..h {
        for px in 0..w {
            let ndc_x = (px as f32) / (w as f32) * 2.0 - 1.0;
            let ndc_y = (py as f32) / (h as f32) * 2.0 - 1.0;

            let (rgb_premul, alpha_frac) = rasterize_pixel(
                ndc_x,
                ndc_y,
                &projected,
                &sun_view_dir,
                theta_min,
                theta_max,
                content_registry,
            );

            if alpha_frac > 1e-6 {
                // Un-premultiply to get true albedo color
                let rgb = [
                    rgb_premul[0] / alpha_frac,
                    rgb_premul[1] / alpha_frac,
                    rgb_premul[2] / alpha_frac,
                ];
                let mapped = aces_tone_map(rgb);
                let r = (linear_to_gamma(mapped[0]) * 255.0).clamp(0.0, 255.0) as u8;
                let g = (linear_to_gamma(mapped[1]) * 255.0).clamp(0.0, 255.0) as u8;
                let b = (linear_to_gamma(mapped[2]) * 255.0).clamp(0.0, 255.0) as u8;
                let a = (alpha_frac * 255.0).clamp(0.0, 255.0) as u8;
                img.put_pixel(px as u32, py as u32, image::Rgba([r, g, b, a]));
            } else {
                img.put_pixel(px as u32, py as u32, image::Rgba([0, 0, 0, 0]));
            }
        }
    }

    img
}

// ─── Stage 1: Transform + Clip + Project ────────────────────────────

fn vertex_shader(
    vert: [f32; 4],
    tex: [f32; 4],
    model_xform: MatN<5>,
    view_matrix: MatN<5>,
    f_xy: f32,
    aspect: f32,
) -> ClipVert {
    let pos5 = VecN::<5>::new([vert[0], vert[1], vert[2], vert[3], 1.0]);
    let view_pos = view_matrix * (model_xform * pos5);

    let depth = (view_pos[2] * view_pos[2] + view_pos[3] * view_pos[3]).sqrt();
    let proj_div = depth / f_xy;

    ClipVert {
        pos: [
            view_pos[0],
            aspect * (-view_pos[1]),
            view_pos[2],
            view_pos[3],
            proj_div,
        ],
        tex,
    }
}

fn clip_edge(a: &ClipVert, b: &ClipVert, da: f32, db: f32) -> ClipVert {
    let t = da / (da - db);
    let omt = 1.0 - t;
    ClipVert {
        pos: [
            a.pos[0] * omt + b.pos[0] * t,
            a.pos[1] * omt + b.pos[1] * t,
            a.pos[2] * omt + b.pos[2] * t,
            a.pos[3] * omt + b.pos[3] * t,
            a.pos[4] * omt + b.pos[4] * t,
        ],
        tex: [
            a.tex[0] * omt + b.tex[0] * t,
            a.tex[1] * omt + b.tex[1] * t,
            a.tex[2] * omt + b.tex[2] * t,
            a.tex[3] * omt + b.tex[3] * t,
        ],
    }
}

fn fix_depth(v: &mut ClipVert, f_xy: f32) {
    v.pos[4] = (v.pos[2] * v.pos[2] + v.pos[3] * v.pos[3]).sqrt() / f_xy;
}

fn clip_tet_generic(tet: &ClipTet, dist: [f32; 4], f_xy: f32, do_fix_depth: bool) -> Vec<ClipTet> {
    let mut inside_idx = Vec::new();
    let mut outside_idx = Vec::new();
    for i in 0..4 {
        if dist[i] >= 0.0 {
            inside_idx.push(i);
        } else {
            outside_idx.push(i);
        }
    }

    let n_in = inside_idx.len();
    if n_in == 4 {
        return vec![tet.clone()];
    }
    if n_in == 0 {
        return vec![];
    }

    let maybe_fix = |mut v: ClipVert| -> ClipVert {
        if do_fix_depth {
            fix_depth(&mut v, f_xy);
        }
        v
    };

    if n_in == 1 {
        let a = inside_idx[0];
        let b = outside_idx[0];
        let c = outside_idx[1];
        let d = outside_idx[2];
        let v_ab = maybe_fix(clip_edge(&tet.verts[a], &tet.verts[b], dist[a], dist[b]));
        let v_ac = maybe_fix(clip_edge(&tet.verts[a], &tet.verts[c], dist[a], dist[c]));
        let v_ad = maybe_fix(clip_edge(&tet.verts[a], &tet.verts[d], dist[a], dist[d]));
        vec![ClipTet {
            verts: [tet.verts[a].clone(), v_ab, v_ac, v_ad],
            material_id: tet.material_id,
        }]
    } else if n_in == 2 {
        let a = inside_idx[0];
        let b = inside_idx[1];
        let c = outside_idx[0];
        let d = outside_idx[1];
        let v_ac = maybe_fix(clip_edge(&tet.verts[a], &tet.verts[c], dist[a], dist[c]));
        let v_ad = maybe_fix(clip_edge(&tet.verts[a], &tet.verts[d], dist[a], dist[d]));
        let v_bc = maybe_fix(clip_edge(&tet.verts[b], &tet.verts[c], dist[b], dist[c]));
        let v_bd = maybe_fix(clip_edge(&tet.verts[b], &tet.verts[d], dist[b], dist[d]));
        vec![
            ClipTet {
                verts: [
                    tet.verts[a].clone(),
                    tet.verts[b].clone(),
                    v_ac.clone(),
                    v_ad.clone(),
                ],
                material_id: tet.material_id,
            },
            ClipTet {
                verts: [
                    tet.verts[b].clone(),
                    v_ac.clone(),
                    v_ad.clone(),
                    v_bc.clone(),
                ],
                material_id: tet.material_id,
            },
            ClipTet {
                verts: [tet.verts[b].clone(), v_ad, v_bc, v_bd],
                material_id: tet.material_id,
            },
        ]
    } else {
        // n_in == 3
        let a = inside_idx[0];
        let b = inside_idx[1];
        let c = inside_idx[2];
        let d = outside_idx[0];
        let v_ad = maybe_fix(clip_edge(&tet.verts[a], &tet.verts[d], dist[a], dist[d]));
        let v_bd = maybe_fix(clip_edge(&tet.verts[b], &tet.verts[d], dist[b], dist[d]));
        let v_cd = maybe_fix(clip_edge(&tet.verts[c], &tet.verts[d], dist[c], dist[d]));
        vec![
            ClipTet {
                verts: [
                    tet.verts[a].clone(),
                    tet.verts[b].clone(),
                    tet.verts[c].clone(),
                    v_ad.clone(),
                ],
                material_id: tet.material_id,
            },
            ClipTet {
                verts: [
                    tet.verts[b].clone(),
                    tet.verts[c].clone(),
                    v_ad.clone(),
                    v_bd.clone(),
                ],
                material_id: tet.material_id,
            },
            ClipTet {
                verts: [tet.verts[c].clone(), v_ad, v_bd, v_cd],
                material_id: tet.material_id,
            },
        ]
    }
}

fn clip_and_project(
    instances: &[ModelInstance],
    model_tets: &[ModelTetrahedron],
    params: &CpuRenderParams,
) -> Vec<ProjectedTet> {
    let f_xy = params.focal_length_xy;
    let aspect = 1.0; // 1:1 for debug images

    // ZW cone angles
    let zw_view_angle = (PI / 2.0) / params.focal_length_zw;
    let theta_min = PI / 4.0 - zw_view_angle / 2.0;
    let theta_max = PI / 4.0 + zw_view_angle / 2.0;
    let sin_min = theta_min.sin();
    let cos_min = theta_min.cos();
    let sin_max = theta_max.sin();
    let cos_max = theta_max.cos();

    let mut result = Vec::new();

    for inst in instances {
        for mt in model_tets {
            let tex_id = inst.cell_material_ids[mt.cell_id as usize];
            if tex_id == 0 {
                continue;
            }

            // Transform vertices
            let initial = ClipTet {
                verts: [
                    vertex_shader(
                        vecn4_to_arr(mt.vertex_positions[0]),
                        vecn4_to_arr(mt.texture_positions[0]),
                        inst.model_transform,
                        params.view_matrix,
                        f_xy,
                        aspect,
                    ),
                    vertex_shader(
                        vecn4_to_arr(mt.vertex_positions[1]),
                        vecn4_to_arr(mt.texture_positions[1]),
                        inst.model_transform,
                        params.view_matrix,
                        f_xy,
                        aspect,
                    ),
                    vertex_shader(
                        vecn4_to_arr(mt.vertex_positions[2]),
                        vecn4_to_arr(mt.texture_positions[2]),
                        inst.model_transform,
                        params.view_matrix,
                        f_xy,
                        aspect,
                    ),
                    vertex_shader(
                        vecn4_to_arr(mt.vertex_positions[3]),
                        vecn4_to_arr(mt.texture_positions[3]),
                        inst.model_transform,
                        params.view_matrix,
                        f_xy,
                        aspect,
                    ),
                ],
                material_id: tex_id,
            };

            // Pass 1: ZW cone lower boundary — dist = -sin(θ_min)*Z + cos(θ_min)*W
            let mut buf: Vec<ClipTet> = vec![initial];
            buf = clip_pass(
                &buf,
                |v| -sin_min * v.pos[2] + cos_min * v.pos[3],
                f_xy,
                false,
            );
            if buf.is_empty() {
                continue;
            }

            // Pass 2: ZW cone upper boundary — dist = sin(θ_max)*Z - cos(θ_max)*W
            buf = clip_pass(
                &buf,
                |v| sin_max * v.pos[2] - cos_max * v.pos[3],
                f_xy,
                false,
            );
            if buf.is_empty() {
                continue;
            }

            // Recompute projDiv from actual Z,W
            for ct in &mut buf {
                for v in &mut ct.verts {
                    fix_depth(v, f_xy);
                }
            }

            // Pass 3: Near depth clip — dist = projDiv - MIN_DEPTH_DIVISOR
            buf = clip_pass(&buf, |v| v.pos[4] - MIN_DEPTH_DIVISOR, f_xy, true);
            if buf.is_empty() {
                continue;
            }

            // Project and emit
            for ct in &buf {
                let pre_proj: [[f32; 4]; 4] = [
                    [
                        ct.verts[0].pos[0],
                        ct.verts[0].pos[1],
                        ct.verts[0].pos[2],
                        ct.verts[0].pos[3],
                    ],
                    [
                        ct.verts[1].pos[0],
                        ct.verts[1].pos[1],
                        ct.verts[1].pos[2],
                        ct.verts[1].pos[3],
                    ],
                    [
                        ct.verts[2].pos[0],
                        ct.verts[2].pos[1],
                        ct.verts[2].pos[2],
                        ct.verts[2].pos[3],
                    ],
                    [
                        ct.verts[3].pos[0],
                        ct.verts[3].pos[1],
                        ct.verts[3].pos[2],
                        ct.verts[3].pos[3],
                    ],
                ];
                let normal = normal4d(&pre_proj);

                let mut pt = ProjectedTet {
                    screen_xy: [[0.0; 2]; 4],
                    zw: [[0.0; 2]; 4],
                    tex: [[0.0; 4]; 4],
                    inv_proj_div: [0.0; 4],
                    normal,
                    material_id: ct.material_id,
                };

                for i in 0..4 {
                    let pd = ct.verts[i].pos[4];
                    pt.screen_xy[i] = [ct.verts[i].pos[0] / pd, ct.verts[i].pos[1] / pd];
                    pt.zw[i] = [ct.verts[i].pos[2], ct.verts[i].pos[3]];
                    pt.tex[i] = ct.verts[i].tex;
                    pt.inv_proj_div[i] = 1.0 / pd;
                }

                result.push(pt);
            }
        }
    }

    result
}

fn clip_pass<F: Fn(&ClipVert) -> f32>(
    tets: &[ClipTet],
    dist_fn: F,
    f_xy: f32,
    do_fix_depth: bool,
) -> Vec<ClipTet> {
    let mut out = Vec::new();
    for t in tets {
        let dist = [
            dist_fn(&t.verts[0]),
            dist_fn(&t.verts[1]),
            dist_fn(&t.verts[2]),
            dist_fn(&t.verts[3]),
        ];
        out.extend(clip_tet_generic(t, dist, f_xy, do_fix_depth));
    }
    out
}

// ─── Stage 2: Per-pixel rasterization ───────────────────────────────

fn rasterize_pixel(
    ndc_x: f32,
    ndc_y: f32,
    tets: &[ProjectedTet],
    sun_view_dir: &[f32; 4],
    theta_min: f32,
    theta_max: f32,
    content_registry: &polychora::content_registry::ContentRegistry,
) -> ([f32; 3], f32) {
    let mut zw_lines: Vec<ZWLine> = Vec::new();
    let point = [ndc_x, ndc_y];

    for tet in tets {
        // Bounding box test
        let mut bb_min = [f32::MAX; 2];
        let mut bb_max = [f32::MIN; 2];
        for i in 0..4 {
            for d in 0..2 {
                bb_min[d] = bb_min[d].min(tet.screen_xy[i][d]);
                bb_max[d] = bb_max[d].max(tet.screen_xy[i][d]);
            }
        }
        if ndc_x < bb_min[0] || ndc_x > bb_max[0] || ndc_y < bb_min[1] || ndc_y > bb_max[1] {
            continue;
        }

        // Test each face
        let mut barys = [[0.0f32; 3]; 4];
        let mut in_tri = [false; 4];
        let mut any_in = false;

        for j in 0..4 {
            let tri = [
                tet.screen_xy[COMBOS[j][0]],
                tet.screen_xy[COMBOS[j][1]],
                tet.screen_xy[COMBOS[j][2]],
            ];
            barys[j] = barycentric_2d(&tri, &point);
            in_tri[j] = barys[j][0] >= 0.0 && barys[j][1] >= 0.0 && barys[j][2] >= 0.0;
            any_in = any_in || in_tri[j];
        }

        if !any_in {
            continue;
        }

        // Find the two intersected faces
        let mut tri_indices = [0usize; 2];
        if in_tri[0] {
            tri_indices[0] = 0;
            tri_indices[1] = if in_tri[1] {
                1
            } else if in_tri[2] {
                2
            } else {
                3
            };
        } else if in_tri[1] {
            tri_indices[0] = 1;
            tri_indices[1] = if in_tri[2] { 2 } else { 3 };
        } else {
            tri_indices[0] = 2;
            tri_indices[1] = 3;
        }

        // Compute ZW line endpoints with perspective-correct interpolation
        let mut line = ZWLine {
            zw: [[0.0; 2]; 2],
            tex: [[0.0; 4]; 2],
            normal: tet.normal,
            material_id: tet.material_id,
        };

        for j in 0..2 {
            let bary = barys[tri_indices[j]];
            let combo = COMBOS[tri_indices[j]];

            let inv_d = [
                tet.inv_proj_div[combo[0]],
                tet.inv_proj_div[combo[1]],
                tet.inv_proj_div[combo[2]],
            ];

            // Interpolate 1/d in screen space
            let interp_inv_d = bary[0] * inv_d[0] + bary[1] * inv_d[1] + bary[2] * inv_d[2];

            // Perspective-correct ZW interpolation
            let mut zw_over_d = [0.0f32; 2];
            for k in 0..2 {
                zw_over_d[k] = bary[0] * tet.zw[combo[0]][k] * inv_d[0]
                    + bary[1] * tet.zw[combo[1]][k] * inv_d[1]
                    + bary[2] * tet.zw[combo[2]][k] * inv_d[2];
            }
            line.zw[j] = [zw_over_d[0] / interp_inv_d, zw_over_d[1] / interp_inv_d];

            // Perspective-correct texture interpolation
            let mut tex_over_d = [0.0f32; 4];
            for k in 0..4 {
                tex_over_d[k] = bary[0] * tet.tex[combo[0]][k] * inv_d[0]
                    + bary[1] * tet.tex[combo[1]][k] * inv_d[1]
                    + bary[2] * tet.tex[combo[2]][k] * inv_d[2];
            }
            line.tex[j] = [
                tex_over_d[0] / interp_inv_d,
                tex_over_d[1] / interp_inv_d,
                tex_over_d[2] / interp_inv_d,
                tex_over_d[3] / interp_inv_d,
            ];
        }

        zw_lines.push(line);
    }

    if zw_lines.is_empty() {
        return ([0.0; 3], 0.0);
    }

    render_zw_lines_simple(&zw_lines, theta_min, theta_max, sun_view_dir, content_registry)
}

// ─── ZW line rendering ──────────────────────────────────────────────

fn render_zw_lines_simple(
    lines: &[ZWLine],
    theta_min: f32,
    theta_max: f32,
    sun_view_dir: &[f32; 4],
    content_registry: &polychora::content_registry::ContentRegistry,
) -> ([f32; 3], f32) {
    let angle_step = (theta_max - theta_min) / DEPTH_FACTOR as f32;

    // Precompute per-line constants
    let precomputed: Vec<_> = lines
        .iter()
        .map(|l| {
            let q0 = l.zw[0];
            let q1 = l.zw[1];
            let dx = q1[0] - q0[0];
            let dy = q1[1] - q0[1];
            let r_num = q0[0] * dy - q0[1] * dx;
            (dx, dy, r_num, q0[0], q0[1])
        })
        .collect();

    // Incremental sin/cos rotation
    let theta0 = theta_min;
    let mut ray_x = theta0.cos();
    let mut ray_y = theta0.sin();
    let cos_step = angle_step.cos();
    let sin_step = angle_step.sin();

    let mut accum = [0.0f32; 3];
    let mut alpha_accum = 0.0f32;

    for _ in 0..DEPTH_FACTOR {
        let mut closest_line: i32 = -1;
        let mut closest_dist = f32::MAX;
        let mut closest_s = 0.0f32;

        for (j, &(dx, dy, r_num, q0x, q0y)) in precomputed.iter().enumerate() {
            let r_den = ray_x * dy - ray_y * dx;
            if r_den.abs() < 1e-12 {
                continue;
            }
            let inv_r_den = 1.0 / r_den;
            let r = r_num * inv_r_den;
            let s = (q0x * ray_y - q0y * ray_x) * inv_r_den;

            if s < 0.0 || s > 1.0 || r < 0.0 {
                continue;
            }

            if closest_line < 0 || r < closest_dist {
                closest_line = j as i32;
                closest_dist = r;
                closest_s = s;
            }
        }

        if closest_line >= 0 {
            let j = closest_line as usize;
            let val = closest_s;
            let tex = [
                lines[j].tex[0][0] * (1.0 - val) + lines[j].tex[1][0] * val,
                lines[j].tex[0][1] * (1.0 - val) + lines[j].tex[1][1] * val,
                lines[j].tex[0][2] * (1.0 - val) + lines[j].tex[1][2] * val,
                lines[j].tex[0][3] * (1.0 - val) + lines[j].tex[1][3] * val,
            ];
            let (albedo, luminance) =
                sample_material(lines[j].material_id, [tex[0], tex[1], tex[2], tex[3]], content_registry);
            let normal = lines[j].normal;

            // Two-sided Lambert diffuse
            let n_dot_l = dot4(&normal, sun_view_dir).max(0.0);
            let n_dot_l_back = (-dot4(&normal, sun_view_dir)).max(0.0);
            let diffuse = n_dot_l.max(n_dot_l_back);

            let ambient = [0.08, 0.09, 0.12];
            let sun_color = [0.8, 0.76, 0.72]; // 1.0*0.8, 0.95*0.8, 0.9*0.8

            for c in 0..3 {
                let lit = albedo[c] * (ambient[c] + sun_color[c] * diffuse) + luminance * albedo[c];
                accum[c] += lit / DEPTH_FACTOR as f32;
            }
            alpha_accum += 1.0 / DEPTH_FACTOR as f32;
        }

        // Rotate ray
        let new_x = ray_x * cos_step - ray_y * sin_step;
        let new_y = ray_x * sin_step + ray_y * cos_step;
        ray_x = new_x;
        ray_y = new_y;
    }

    (accum, alpha_accum)
}

// ─── Material sampling ──────────────────────────────────────────────

fn fract(x: f32) -> f32 {
    x - x.floor()
}

fn saturate(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = saturate((x - edge0) / (edge1 - edge0));
    t * t * (3.0 - 2.0 * t)
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn sample_material(id: u32, tex_pos: [f32; 4], content_registry: &polychora::content_registry::ContentRegistry) -> ([f32; 3], f32) {
    const TAU: f32 = 6.283_185_5;
    let basic_lum = 0.001;
    let p = [
        fract(tex_pos[0]),
        fract(tex_pos[1]),
        fract(tex_pos[2]),
        fract(tex_pos[3]),
    ];
    match id {
        1 => ([1.0, 0.0, 0.0], basic_lum), // Red
        2 => ([1.0, 0.8, 0.0], basic_lum), // Orange
        3 => ([0.5, 1.0, 0.0], basic_lum), // Yellow-green
        4 => ([0.0, 1.0, 0.2], basic_lum), // Green
        5 => ([0.0, 1.0, 1.0], basic_lum), // Cyan
        6 => ([0.0, 0.2, 1.0], basic_lum), // Blue
        7 => ([0.5, 0.0, 1.0], basic_lum), // Purple
        8 => ([1.0, 0.0, 0.8], basic_lum), // Magenta
        9 => {
            let albedo = [
                (tex_pos[0] + 1.0) / 2.0,
                (tex_pos[1] + 1.0) / 2.0,
                (tex_pos[2] + 1.0) / 2.0,
            ];
            (albedo, 0.4)
        }
        10 => ([39.0 / 256.0, 69.0 / 256.0, 19.8 / 256.0], 0.0), // Brown
        11 => {
            // Neutral 4D ground grid
            let minor = [
                (fract(tex_pos[0] + 0.5) - 0.5).abs(),
                (fract(tex_pos[1] + 0.5) - 0.5).abs(),
                (fract(tex_pos[2] + 0.5) - 0.5).abs(),
                (fract(tex_pos[3] + 0.5) - 0.5).abs(),
            ];
            let minor_axis = minor[0].min(minor[1]).min(minor[2]).min(minor[3]);
            let minor_line = 1.0 - saturate(minor_axis / 0.055);

            let major = [
                (fract(tex_pos[0] * 0.25 + 0.5) - 0.5).abs(),
                (fract(tex_pos[1] * 0.25 + 0.5) - 0.5).abs(),
                (fract(tex_pos[2] * 0.25 + 0.5) - 0.5).abs(),
                (fract(tex_pos[3] * 0.25 + 0.5) - 0.5).abs(),
            ];
            let major_axis = major[0].min(major[1]).min(major[2]).min(major[3]);
            let major_line = 1.0 - saturate(major_axis / 0.085);

            let base = [0.45, 0.47, 0.49];
            let minor_col = [0.36, 0.38, 0.40];
            let major_col = [0.66, 0.68, 0.71];
            let mut col = lerp3(base, minor_col, minor_line * 0.6);
            col = lerp3(col, major_col, major_line);
            (col, 0.0)
        }
        12 => ([1.0, 1.0, 1.0], 0.0),  // White
        13 => ([1.0, 1.0, 1.0], 40.0), // Light source
        14 => ([1.0, 1.0, 1.0], 0.0),  // Mirror walls
        15 => {
            // Lava-veined basalt
            let vein_field = 0.5 + 0.5 * ((p[0] * 11.0 + p[1] * 13.0 + p[2] * 17.0) * TAU).sin();
            let veins = vein_field.powf(6.0);
            let rock_color = [0.08, 0.05, 0.04];
            let lava_color = [0.95, 0.36, 0.08];
            (lerp3(rock_color, lava_color, veins), 0.08 + 6.0 * veins)
        }
        16 => {
            // Crystal lattice
            let c0 = (fract(p[0] * 8.0) - 0.5).abs();
            let c1 = (fract(p[1] * 8.0) - 0.5).abs();
            let c2 = (fract(p[2] * 8.0) - 0.5).abs();
            let line = 1.0 - saturate(c0.min(c1).min(c2) * 36.0);
            let base_color = [0.02, 0.06, 0.09];
            let line_color = [0.25, 0.9, 1.0];
            (lerp3(base_color, line_color, line), 0.2 + 2.2 * line)
        }
        17 => {
            // Marble swirl
            let swirl = ((p[0] * 14.0 + p[2] * 9.0) * TAU + (p[1] * TAU * 5.0).sin() * 1.5).sin();
            let veins = smoothstep(0.62, 0.88, 0.5 + 0.5 * swirl);
            let stone_base = [0.82, 0.84, 0.88];
            let vein_color = [0.96, 0.97, 1.0];
            (lerp3(stone_base, vein_color, veins), 0.0)
        }
        18 => {
            // Oxidized metal
            let dot_v = (p[0] * 16.0) * 12.9898 + (p[1] * 16.0) * 78.233 + (p[2] * 16.0) * 37.719;
            let noise = fract(dot_v.sin() * 43_758.547);
            let rust = smoothstep(0.42, 0.8, noise);
            let steel = [0.55, 0.58, 0.60];
            let oxide = [0.48, 0.18, 0.07];
            (lerp3(steel, oxide, rust), 0.0)
        }
        19 => {
            // Bio-spore moss
            let cell = [
                (p[0] * 10.0).floor(),
                (p[1] * 10.0).floor(),
                (p[2] * 10.0).floor(),
            ];
            let seed =
                fract((cell[0] * 17.13 + cell[1] * 3.71 + cell[2] * 29.97).sin() * 15_731.743);
            let spores = smoothstep(0.93, 1.0, seed);
            let moss_base = [0.06, 0.22, 0.09];
            let spore_glow = [0.25, 0.75, 0.35];
            (lerp3(moss_base, spore_glow, spores), spores * 3.5)
        }
        20 => {
            // Void mirror
            let q = [p[0] - 0.5, p[1] - 0.5, p[2] - 0.5];
            let q_len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
            let center_glow = saturate(1.0 - q_len * 1.8).powf(2.0);
            (
                [
                    0.03 + 0.20 * center_glow,
                    0.05 + 0.10 * center_glow,
                    0.12 + 0.20 * center_glow,
                ],
                0.15 * center_glow,
            )
        }
        _ => {
            let [r, g, b] = content_registry.material_color_by_token(id.min(u16::MAX as u32) as u16);
            ([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0], 0.0)
        }
    }
}

// ─── Tone mapping + gamma ───────────────────────────────────────────

fn aces_tone_map(rgb: [f32; 3]) -> [f32; 3] {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    rgb.map(|x| ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0))
}

fn linear_to_gamma(value: f32) -> f32 {
    value.powf(1.0 / 2.2)
}

// ─── Math helpers ───────────────────────────────────────────────────

fn barycentric_2d(tri: &[[f32; 2]; 3], point: &[f32; 2]) -> [f32; 3] {
    // Uses cross product method matching GPU shader (getBarycentric)
    let u = cross3(
        [
            tri[2][0] - tri[0][0],
            tri[1][0] - tri[0][0],
            tri[0][0] - point[0],
        ],
        [
            tri[2][1] - tri[0][1],
            tri[1][1] - tri[0][1],
            tri[0][1] - point[1],
        ],
    );

    if u[2].abs() < 1e-6 {
        return [-1.0, 1.0, 1.0];
    }

    [1.0 - (u[0] + u[1]) / u[2], u[1] / u[2], u[0] / u[2]]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normal4d(verts: &[[f32; 4]; 4]) -> [f32; 4] {
    let e1 = sub4(verts[1], verts[0]);
    let e2 = sub4(verts[2], verts[0]);
    let e3 = sub4(verts[3], verts[0]);

    let n = cross4d(e1, e2, e3);
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2] + n[3] * n[3]).sqrt();
    if len > 1e-6 {
        [n[0] / len, n[1] / len, n[2] / len, n[3] / len]
    } else {
        [0.0; 4]
    }
}

fn cross4d(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
    // 4D cross product via cofactor expansion of:
    // | e1  e2  e3  e4 |
    // | ax  ay  az  aw |
    // | bx  by  bz  bw |
    // | cx  cy  cz  cw |
    [
        det3x3(a[1], a[2], a[3], b[1], b[2], b[3], c[1], c[2], c[3]),
        -det3x3(a[0], a[2], a[3], b[0], b[2], b[3], c[0], c[2], c[3]),
        det3x3(a[0], a[1], a[3], b[0], b[1], b[3], c[0], c[1], c[3]),
        -det3x3(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]),
    ]
}

fn det3x3(
    a00: f32,
    a01: f32,
    a02: f32,
    a10: f32,
    a11: f32,
    a12: f32,
    a20: f32,
    a21: f32,
    a22: f32,
) -> f32 {
    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
}

fn sub4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

fn dot4(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn vecn4_to_arr(v: impl Into<VecN<4>>) -> [f32; 4] {
    let v: VecN<4> = v.into();
    [v[0], v[1], v[2], v[3]]
}
