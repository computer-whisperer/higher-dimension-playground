

use core::array;
use std::marker::PhantomData;

// compiler go brr
pub const fn factorial(n: usize) -> usize {
    if n == 0 || n == 1 {
        1
    } else {
        n * factorial(n-1)
    }
}

pub const fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        0
    } else {
        factorial(n) / (factorial(k) * factorial(n-k))
    }
}


const fn generate_combinations<const N: usize, const K: usize>() -> [[usize; K]; binomial(N, K)] {
    let mut combinations = [[0; K]; binomial(N, K)];
    let mut current_combination_index = 0;

    // Initial combination starts with 0, 1, 2, ..., K-1
    let mut current: [usize; K] = {
        let mut init = [0; K];
        for i in 0..K {
            init[i] = i;
        }
        init
    };

    while current[0] <= N - K {
        // Store the current combination
        combinations[current_combination_index] = current;
        current_combination_index += 1;

        // Generate next combination
        let mut j = K - 1;
        while j > 0 && current[j] == N - K + j {
            j -= 1;
        }

        current[j] += 1;
        for k in j+1..K {
            current[k] = current[k-1] + 1;
        }
    }

    combinations
}

/// Generates all permutations of indices 0..K at compile-time
///
/// # Type Parameters
/// - `K`: The number of elements to generate permutations for
/// - `const N: usize`: The total number of permutations (K!)
///
/// # Returns
/// A const array containing all possible permutations
const fn generate_permutations<const K: usize>() -> [[usize; K]; factorial(K)] {
    let mut permutations = [[0; K]; factorial(K)];
    let mut current_perm = [0; K];

    // Initialize first permutation as 0, 1, 2, ...
    for i in 0..K {
        current_perm[i] = i;
    }

    let mut perm_index = 0;

    loop {
        // Store current permutation
        permutations[perm_index] = current_perm;
        perm_index += 1;

        // Generate next permutation using Heap's algorithm
        if !next_permutation(&mut current_perm) {
            break;
        }
    }

    permutations
}

/// Generates the next lexicographically greater permutation
///
/// Returns true if a new permutation was generated, false if no more permutations exist
const fn next_permutation(perm: &mut [usize]) -> bool {
    let k = perm.len();

    // Find the largest index i such that perm[i] < perm[i + 1]
    let mut i = k as isize - 2;
    while i >= 0 && perm[i as usize] >= perm[(i + 1) as usize] {
        i -= 1;
    }

    // If no such index exists, we've gone through all permutations
    if i < 0 {
        return false;
    }

    // Find the largest index j > i such that perm[j] > perm[i]
    let mut j = (k - 1) as isize;
    while j > i && perm[j as usize] <= perm[i as usize] {
        j -= 1;
    }

    // Swap elements at indices i and j
    let temp = perm[i as usize];
    perm[i as usize] = perm[j as usize];
    perm[j as usize] = temp;

    // Reverse the subarray to the right of index i
    let mut left = (i + 1) as usize;
    let mut right = k - 1;

    while left < right {
        let temp = perm[left];
        perm[left] = perm[right];
        perm[right] = temp;
        left += 1;
        right -= 1;
    }

    true
}

#[allow(dead_code)]
pub struct Hypercube<const D: usize, T: From<usize>> {
    phantom_data: PhantomData<T>
}

#[allow(dead_code)]
impl<const D: usize, T: From<usize>> Hypercube<D, T> {

    pub(crate) fn generate_vertices() -> [[T; D]; 1 << D] {
        array::from_fn(|i| {
            array::from_fn(|d| {
                ((i>>d)&1).into()
            })
        })
    }
}

/*
#[allow(dead_code)]
impl<const D: usize, T: From<usize>> Hypercube<D, T> {

    // For each dimension, for each old vertex
    fn generate_edges() -> [[usize; 2]; D*(1<<(D-1))] {
        array::from_fn(|i| {
            let dim = i%D;
            let mut id_in_dim = i/D;
            
            let mut base = 0;
            
            for d in 0..D {
                if d == dim {
                    // Leave 0
                }
                else {
                    base |= (id_in_dim%2) << d;
                    id_in_dim = id_in_dim/2;
                }
            }
            
            [base, base | (1<<dim)]
        })
    }
}
*/

pub fn generate_simplexes_for_k_face<const D: usize, const K: usize>(k_face: [usize; K+1]) -> [[usize; K+1]; factorial(K)] {
    let permutations = generate_permutations::<K>();

    array::from_fn(|i|{
        let permutation = permutations[i];
        let mut output = [0; K+1];

        let mut accumulation = k_face[0];

        output[0] = accumulation;
        for j in 0..K {
            accumulation |= k_face[permutation[j] + 1];
            output[j + 1] = accumulation;
        }

        output
    })
}

#[allow(dead_code)]
impl<const D: usize, T: From<usize>> Hypercube<D, T> {
    
    pub fn generate_k_faces<const K: usize>() -> [[usize; K+1]; (1<<(D - K))*binomial(D, K)] {
        let combo = generate_combinations::<D, K>();
        
        array::from_fn(|i| {
            let mod_id = i%combo.len();
            let mut base_id = i/combo.len();
            
            let masked_bits = combo[mod_id];
            
            let mut base= 0;
            for d in 0..D {
                if masked_bits.contains(&d) {
                    // Leave 0
                }
                else {
                    // Set from base id
                    base |= (base_id%2) << d;
                    base_id = base_id/2;
                }
            }
            
            let mut output = [base; K+1];
            
            for i in 0..K {
                output[i + 1] |= 1<<masked_bits[i];
            }
            
            output
        })
    }
    
    pub fn generate_simplexes<const K: usize>() -> [[usize; K+1]; (1<<(D - K))*binomial(D, K)*factorial(K)] {
        let faces = Self::generate_k_faces::<K>();
        let simplexes = faces.map(|face| {generate_simplexes_for_k_face::<D, K>(face)});
        let simplexes_per_face = factorial(K);
        array::from_fn(|i|{
            simplexes[i/simplexes_per_face][i%simplexes_per_face]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(10, 5), 252);
        assert_eq!(binomial(10, 5), 252);
    }

    #[test]
    fn test_combinations() {
        assert_eq!(&generate_combinations::<2, 1>(), &[[0], [1]]);
        assert_eq!(&generate_combinations::<3, 2>(), &[[0, 1], [0, 2], [1, 2]]);
    }

    #[test]
    fn test_permutations() {
        assert_eq!(&generate_permutations::<2>(), &[[0, 1], [1, 0]]);
        assert_eq!(&generate_permutations::<3>(), &[[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]);
    }

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
/*
    #[test]
    fn test_hypercube_edges() {
        assert_eq!(&Hypercube::<2, usize>::generate_edges(), &[
            [0b00, 0b01], [0b00, 0b10], 
            [0b10, 0b11], [0b01, 0b11]
        ]);
        assert_eq!(&Hypercube::<3, usize>::generate_edges(), &[
            [0b000, 0b001], [0b000, 0b010], [0b000, 0b100], 
            [0b010, 0b011], [0b001, 0b011], [0b001, 0b101],
            [0b100, 0b101], [0b100, 0b110], [0b010, 0b110],
            [0b110, 0b111], [0b101, 0b111], [0b011, 0b111]
        ]);
    }
*/
    #[test]
    fn test_2_faces() {
        assert_eq!(&Hypercube::<2, usize>::generate_k_faces::<1>(), &[
            [0b00, 0b01], [0b00, 0b10],
            [0b10, 0b11], [0b01, 0b11]
        ]);
        assert_eq!(&Hypercube::<3, usize>::generate_k_faces::<1>(), &[
            [0b000, 0b001], [0b000, 0b010], [0b000, 0b100],
            [0b010, 0b011], [0b001, 0b011], [0b001, 0b101],
            [0b100, 0b101], [0b100, 0b110], [0b010, 0b110],
            [0b110, 0b111], [0b101, 0b111], [0b011, 0b111]
        ]);
        assert_eq!(&Hypercube::<2, usize>::generate_k_faces::<2>(), &[
            [0b00, 0b01, 0b10]
        ]);
        assert_eq!(&Hypercube::<3, usize>::generate_k_faces::<2>(), &[
            [0b000, 0b001, 0b010],
            [0b000, 0b001, 0b100],
            [0b000, 0b010, 0b100],
            [0b100, 0b101, 0b110],
            [0b010, 0b011, 0b110],
            [0b001, 0b011, 0b101]
        ]);
    }
    
    #[test]
    fn test_simplexes() {
        assert_eq!(&Hypercube::<2, usize>::generate_simplexes::<1>(), &[
            [0b00, 0b01], [0b00, 0b10], [0b10, 0b11], [0b01, 0b11]
        ]);
        assert_eq!(&Hypercube::<2, usize>::generate_simplexes::<2>(), &[
            [0b00, 0b01, 0b11], [0b00, 0b10, 0b11]
        ]);
        assert_eq!(&Hypercube::<3, usize>::generate_simplexes::<1>(), &[
            [0b000, 0b001], [0b000, 0b010], [0b000, 0b100],
            [0b010, 0b011], [0b001, 0b011], [0b001, 0b101],
            [0b100, 0b101], [0b100, 0b110], [0b010, 0b110],
            [0b110, 0b111], [0b101, 0b111], [0b011, 0b111]
        ]);
        assert_eq!(&Hypercube::<3, usize>::generate_simplexes::<2>(), &[
            [0b000, 0b001, 0b011], [0b000, 0b010, 0b011],
            [0b000, 0b001, 0b101], [0b000, 0b100, 0b101],
            [0b000, 0b010, 0b110], [0b000, 0b100, 0b110],
            [0b100, 0b101, 0b111], [0b100, 0b110, 0b111],
            [0b010, 0b011, 0b111], [0b010, 0b110, 0b111],
            [0b001, 0b011, 0b111], [0b001, 0b101, 0b111]
        ]);
        assert_eq!(&Hypercube::<4, usize>::generate_simplexes::<1>(), &[
            [0b0000, 0b0001], [0b0000, 0b0010], [0b0000, 0b0100], [0b0000, 0b1000],
            [0b0010, 0b0011], [0b0001, 0b0011], [0b0001, 0b0101], [0b0001, 0b1001],
            [0b0100, 0b0101], [0b0100, 0b0110], [0b0010, 0b0110], [0b0010, 0b1010],
            [0b0110, 0b0111], [0b0101, 0b0111], [0b0011, 0b0111], [0b0011, 0b1011],
            
            [0b1000, 0b1001], [0b1000, 0b1010], [0b1000, 0b1100], [0b0100, 0b1100],
            [0b1010, 0b1011], [0b1001, 0b1011], [0b1001, 0b1101], [0b0101, 0b1101],
            [0b1100, 0b1101], [0b1100, 0b1110], [0b1010, 0b1110], [0b0110, 0b1110],
            [0b1110, 0b1111], [0b1101, 0b1111], [0b1011, 0b1111], [0b0111, 0b1111]
        ]);
    }
}