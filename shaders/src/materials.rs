use glam::{Vec3, vec3, Vec4};

pub struct MaterialProperties {
    pub albedo: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub luminance: f32,
    pub padding: u32
}

impl MaterialProperties {
    fn new(albedo: Vec3, metallic: f32, roughness: f32, luminance: f32) -> Self {
        Self {
            albedo: Vec4::new(albedo.x, albedo.y, albedo.z, 1.0),
            metallic,
            roughness,
            luminance,
            padding: 0
        }
    }
}

pub fn sample_material(texture_id: u32, texture_pos: Vec3) -> MaterialProperties {
    match texture_id {
        1 => {MaterialProperties::new(vec3(1.0, 0.0, 0.0), 0.0, 0.0, 0.0)}
        2 => {MaterialProperties::new(vec3(1.0, 0.8, 0.0), 0.0, 0.0, 0.0)}
        3 => {MaterialProperties::new(vec3(0.5, 1.0, 0.0), 0.0, 0.0, 0.0)}
        4 => {MaterialProperties::new(vec3(0.0, 1.0, 0.2), 0.0, 0.0, 0.0)}
        5 => {MaterialProperties::new(vec3(0.0, 1.0, 1.0), 0.0, 0.0, 0.0)}
        6 => {MaterialProperties::new(vec3(0.0, 0.2, 1.0), 0.0, 0.0, 0.0)}
        7 => {MaterialProperties::new(vec3(0.5, 0.0, 1.0), 0.0, 0.0, 0.0)}
        8 => {MaterialProperties::new(vec3(1.0, 0.0, 0.8), 0.0, 0.0, 0.0)}
        9 => {MaterialProperties::new((texture_pos+vec3(1.0, 1.0, 1.0))/2.0, 0.0, 0.0, 0.0)}
        10 => {MaterialProperties::new(vec3(39.0, 69.0, 19.8)/256.0, 0.0, 0.0, 0.0)}
        11 => {MaterialProperties::new(vec3(34.0/256.0, 139.0/256.0, 34.0/256.0)*0.4, 0.0, 0.3, 0.0)}
        12 => {MaterialProperties::new(vec3(1.0, 1.0, 1.0), 0.0, 0.0, 0.0)}
        13 => {MaterialProperties::new(vec3(0.2, 0.2, 1.0), 0.0, 0.0, 10.0)}
        14 => {MaterialProperties::new(vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0)}
        _ => {MaterialProperties::new(vec3(0.0, 0.0, 0.0), 0.0, 0.0, 0.0)}
    }

    /*
            vec3<f32>(1.0, 0.0, 0.0), //0
        vec3<f32>(1.0, 0.0, 0.0), //1
        vec3<f32>(1.0, 0.8, 0.0), //2
        vec3<f32>(0.5, 1.0, 0.0), //3
        vec3<f32>(0.0, 1.0, 0.2), //4
        vec3<f32>(0.0, 1.0, 1.0), //5
        vec3<f32>(0.0, 0.2, 1.0), //6
        vec3<f32>(0.5, 0.0, 1.0), //7
        vec3<f32>(1.0, 0.0, 0.8), //8
        (texture_pos+vec3<f32>(1.0, 1.0, 1.0))/2.0,  // 9
        vec3<f32>(139.0, 69.0, 19.8)/256.0, // 10
        vec3<f32>(34.0, 139.0, 34.0)/256.0, // 11
        vec3<f32>(0.0, 0.0, 1.0), // 12
    */
}