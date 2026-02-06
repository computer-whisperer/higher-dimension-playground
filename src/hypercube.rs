use std::marker::PhantomData;
use common::{factorial, binomial, generate_combinations, generate_permutations};

#[allow(dead_code)]
pub struct Hypercube<const D: usize, T: From<usize>> {
    phantom_data: PhantomData<T>
}

#[allow(dead_code)]
impl<const D: usize, T: From<usize> + Copy + Default> Hypercube<D, T> {

    pub fn generate_vertices() -> Vec<[T; D]> {
        (0..(1 << D)).map(|i| {
            std::array::from_fn(|d| ((i >> d) & 1).into())
        }).collect()
    }

    /// Generate 1-faces (edges) of a D-dimensional hypercube
    /// Returns Vec of edges, each edge is [base_vertex, other_vertex]
    pub fn generate_k_faces_1() -> Vec<[usize; 2]> {
        let k = 1;
        let combinations = generate_combinations(D, k);
        let num_faces = (1 << (D - k)) * binomial(D, k);
        let mut faces = Vec::with_capacity(num_faces);

        for base_id_init in 0..(1 << (D - k)) {
            for combo in &combinations {
                let mut base_id = base_id_init;
                let mut base = 0usize;

                for d in 0..D {
                    if combo.contains(&d) {
                        // Leave 0
                    } else {
                        base |= (base_id % 2) << d;
                        base_id /= 2;
                    }
                }

                let mut output = [base; 2];
                for i in 0..k {
                    output[i + 1] |= 1 << combo[i];
                }
                faces.push(output);
            }
        }

        faces
    }

    /// Generate 3-faces (cells) of a D-dimensional hypercube
    /// Returns Vec of cells, each cell is [base, v1, v2, v3]
    pub fn generate_k_faces_3() -> Vec<[usize; 4]> {
        let k = 3;
        let combinations = generate_combinations(D, k);
        let num_faces = (1 << (D - k)) * binomial(D, k);
        let mut faces = Vec::with_capacity(num_faces);

        for base_id_init in 0..(1 << (D - k)) {
            for combo in &combinations {
                let mut base_id = base_id_init;
                let mut base = 0usize;

                for d in 0..D {
                    if combo.contains(&d) {
                        // Leave 0
                    } else {
                        base |= (base_id % 2) << d;
                        base_id /= 2;
                    }
                }

                let mut output = [base; 4];
                for i in 0..k {
                    output[i + 1] |= 1 << combo[i];
                }
                faces.push(output);
            }
        }

        faces
    }
}

/// Generate all simplexes for a single 3-face (tetrahedron cell)
/// D is the dimension of the hypercube
pub fn generate_simplexes_for_k_face_3<const D: usize>(k_face: [usize; 4]) -> Vec<[usize; 4]> {
    let k = 3;
    let permutations = generate_permutations(k);
    let mut simplexes = Vec::with_capacity(factorial(k));

    for perm in permutations {
        let mut output = [0usize; 4];
        let mut accumulation = k_face[0];

        output[0] = accumulation;
        for j in 0..k {
            accumulation |= k_face[perm[j] + 1];
            output[j + 1] = accumulation;
        }

        simplexes.push(output);
    }

    simplexes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypercube_vertices() {
        let vertices = Hypercube::<3, usize>::generate_vertices();
        assert_eq!(vertices.len(), 8);
        assert_eq!(vertices[0], [0, 0, 0]);
        assert_eq!(vertices[7], [1, 1, 1]);

        let vertices = Hypercube::<4, usize>::generate_vertices();
        assert_eq!(vertices.len(), 16);
        assert_eq!(vertices[0], [0, 0, 0, 0]);
        assert_eq!(vertices[15], [1, 1, 1, 1]);
    }

    #[test]
    fn test_k_faces_1() {
        assert_eq!(Hypercube::<2, usize>::generate_k_faces_1(), vec![
            [0b00, 0b01], [0b00, 0b10],
            [0b10, 0b11], [0b01, 0b11]
        ]);
        assert_eq!(Hypercube::<3, usize>::generate_k_faces_1(), vec![
            [0b000, 0b001], [0b000, 0b010], [0b000, 0b100],
            [0b010, 0b011], [0b001, 0b011], [0b001, 0b101],
            [0b100, 0b101], [0b100, 0b110], [0b010, 0b110],
            [0b110, 0b111], [0b101, 0b111], [0b011, 0b111]
        ]);
    }

    #[test]
    fn test_k_faces_3() {
        assert_eq!(Hypercube::<3, usize>::generate_k_faces_3(), vec![
            [0b000, 0b001, 0b010, 0b100]
        ]);
        assert_eq!(Hypercube::<4, usize>::generate_k_faces_3().len(), 8);
    }

    #[test]
    fn test_simplexes_for_k_face_3() {
        let face = [0b000, 0b001, 0b010, 0b100];
        let simplexes = generate_simplexes_for_k_face_3::<3>(face);
        assert_eq!(simplexes.len(), 6); // 3! = 6 simplexes per 3-face
        assert_eq!(simplexes, vec![
            [0b000, 0b001, 0b011, 0b111],
            [0b000, 0b001, 0b101, 0b111],
            [0b000, 0b010, 0b011, 0b111],
            [0b000, 0b010, 0b110, 0b111],
            [0b000, 0b100, 0b101, 0b111],
            [0b000, 0b100, 0b110, 0b111]
        ]);
    }
}
