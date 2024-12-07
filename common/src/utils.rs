use glam::Vec4;

// compiler go brr
pub const fn factorial(n: usize) -> usize {
    /*
    if n == 0 || n == 1 {
        1
    } else {
        n * factorial(n-1)
    }*/
    match n {
        0 => 1,
        1 => 1,
        2 => 2,
        3 => 6,
        4 => 24,
        5 => 120,
        6 => 720,
        7 => 5040,
        8 => 40320,
        9 => 362880,
        _ => {unimplemented!()}
    }
}

pub const fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        0
    } else {
        factorial(n) / (factorial(k) * factorial(n-k))
    }
}


pub const fn generate_combinations<const N: usize, const K: usize>() -> [[usize; K]; binomial(N, K)] {
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

/// Generates all permutations of indices 0 . . K at compile-time
///
/// # Type Parameters
/// - `K`: The number of elements to generate permutations for
/// - `const N: usize`: The total number of permutations (K!)
///
/// # Returns
/// A const array containing all possible permutations
pub const fn generate_permutations<const K: usize>() -> [[usize; K]; factorial(K)] {
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

const PCG32_DEFAULT_STATE: u64 =  0x853c49e6748fea9b; 
const PCG32_DEFAULT_STREAM: u64 = 0xda3e39cb94b95bdb;
const PCG32_MULT: u64 =        0x5851f42d4c957f2d;

pub struct BasicRNG {
    state: u64,
    inc: u64,
}

impl BasicRNG {
    pub fn new(stream: u64) -> Self {
        Self { 
            state: PCG32_DEFAULT_STATE,
            inc: stream
        }
    }

    // Output value in range [0, 0xFFFFFFFF]
    pub fn rand(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate.wrapping_mul(PCG32_MULT).wrapping_add(self.inc);
        let xorshifted: u32 =  (((oldstate >> 18) ^ oldstate) >> 27) as u32;
        let rot: u32 = (oldstate >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << (((!rot).wrapping_add(1)) & 31))
    }
    
    pub fn rand_f32(&mut self) -> f32 {
        let r = self.rand();
        r as f32 / 0xFFFFFFFFu32 as f32
    }
    
    pub fn rand_vec4(&mut self) -> Vec4 {
        loop {
            let value = Vec4::new(
                self.rand_f32()-0.5,
                self.rand_f32()-0.5,
                self.rand_f32()-0.5,
                self.rand_f32()-0.5
            );
            if value.length() > 1.0 {
                continue; // We're dealing with a unit vector, so continue
            }
            break value.normalize();
        }
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
}