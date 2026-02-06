use glam::Vec4;

/// Factorial function
pub fn factorial(n: usize) -> usize {
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
        _ => (1..=n).product()
    }
}

/// Binomial coefficient C(n, k)
pub fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        0
    } else {
        factorial(n) / (factorial(k) * factorial(n - k))
    }
}

/// Generates all combinations of K elements from N elements
/// Returns Vec of combinations, each combination is a Vec of indices
pub fn generate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut combinations = Vec::with_capacity(binomial(n, k));
    let mut current: Vec<usize> = (0..k).collect();

    loop {
        combinations.push(current.clone());

        // Find rightmost element that can be incremented
        let mut i = k as isize - 1;
        while i >= 0 && current[i as usize] == n - k + i as usize {
            i -= 1;
        }

        if i < 0 {
            break;
        }

        // Increment and reset elements to the right
        current[i as usize] += 1;
        for j in (i as usize + 1)..k {
            current[j] = current[j - 1] + 1;
        }
    }

    combinations
}

/// Generates all permutations of K elements (0..K)
/// Returns Vec of permutations, each permutation is a Vec of indices
pub fn generate_permutations(k: usize) -> Vec<Vec<usize>> {
    let mut permutations = Vec::with_capacity(factorial(k));
    let mut current: Vec<usize> = (0..k).collect();

    loop {
        permutations.push(current.clone());

        if !next_permutation(&mut current) {
            break;
        }
    }

    permutations
}

/// Generates the next lexicographically greater permutation
/// Returns true if a new permutation was generated, false if no more permutations exist
fn next_permutation(perm: &mut [usize]) -> bool {
    let k = perm.len();
    if k <= 1 {
        return false;
    }

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
    perm.swap(i as usize, j as usize);

    // Reverse the subarray to the right of index i
    perm[(i as usize + 1)..].reverse();

    true
}

const PCG32_DEFAULT_STATE: u64 = 0x853c49e6748fea9b;
const PCG32_MULT: u64 = 0x5851f42d4c957f2d;

pub struct BasicRNG {
    state: u64,
    inc: u64,
}

impl BasicRNG {
    pub fn new(stream: u64) -> Self {
        Self {
            state: PCG32_DEFAULT_STATE,
            inc: stream,
        }
    }

    // Output value in range [0, 0xFFFFFFFF]
    pub fn rand(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate.wrapping_mul(PCG32_MULT).wrapping_add(self.inc);
        let xorshifted: u32 = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
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
                self.rand_f32() - 0.5,
                self.rand_f32() - 0.5,
                self.rand_f32() - 0.5,
                self.rand_f32() - 0.5,
            );
            if value.length() > 1.0 {
                continue;
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
        assert_eq!(generate_combinations(2, 1), vec![vec![0], vec![1]]);
        assert_eq!(generate_combinations(3, 2), vec![vec![0, 1], vec![0, 2], vec![1, 2]]);
    }

    #[test]
    fn test_permutations() {
        assert_eq!(generate_permutations(2), vec![vec![0, 1], vec![1, 0]]);
        assert_eq!(
            generate_permutations(3),
            vec![
                vec![0, 1, 2],
                vec![0, 2, 1],
                vec![1, 0, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
                vec![2, 1, 0]
            ]
        );
    }
}
