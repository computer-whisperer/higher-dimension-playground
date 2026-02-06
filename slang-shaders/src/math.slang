// N-dimensional math library for 4D rendering
// Slang supports generics, so we can express VecN<N> patterns

// Generic N-dimensional vector
struct VecN<let N : int>
{
    float data[N];

    __init()
    {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            data[i] = 0.0;
    }

    __init(float v)
    {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            data[i] = v;
    }

    __subscript(int i) -> float
    {
        get { return data[i]; }
        set { data[i] = newValue; }
    }

    float x() { return data[0]; }
    float y() { return data[1]; }
    float z() { return data[2]; }
    float w() { return data[3]; }
    float v() { return data[4]; } // 5th component for homogeneous 4D

    static VecN<N> fromFloat4(float4 v)
    {
        VecN<N> result;
        result[0] = v.x;
        result[1] = v.y;
        if (N > 2) result[2] = v.z;
        if (N > 3) result[3] = v.w;
        return result;
    }

    float4 toFloat4()
    {
        return float4(
            data[0],
            N > 1 ? data[1] : 0.0,
            N > 2 ? data[2] : 0.0,
            N > 3 ? data[3] : 0.0
        );
    }

    VecN<N+1> extend(float val)
    {
        VecN<N+1> result;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result[i] = data[i];
        result[N] = val;
        return result;
    }

    VecN<N-1> truncate()
    {
        VecN<N-1> result;
        [ForceUnroll]
        for (int i = 0; i < N-1; i++)
            result[i] = data[i];
        return result;
    }
}

// Vector operations
VecN<N> operator+<let N : int>(VecN<N> a, VecN<N> b)
{
    VecN<N> result;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        result[i] = a[i] + b[i];
    return result;
}

VecN<N> operator-<let N : int>(VecN<N> a, VecN<N> b)
{
    VecN<N> result;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        result[i] = a[i] - b[i];
    return result;
}

VecN<N> operator*<let N : int>(VecN<N> a, float s)
{
    VecN<N> result;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        result[i] = a[i] * s;
    return result;
}

VecN<N> operator*<let N : int>(float s, VecN<N> a)
{
    return a * s;
}

VecN<N> operator/<let N : int>(VecN<N> a, float s)
{
    return a * (1.0 / s);
}

VecN<N> operator-<let N : int>(VecN<N> a)
{
    VecN<N> result;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        result[i] = -a[i];
    return result;
}

float dot<let N : int>(VecN<N> a, VecN<N> b)
{
    float result = 0.0;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        result += a[i] * b[i];
    return result;
}

float lengthSquared<let N : int>(VecN<N> a)
{
    return dot(a, a);
}

float length<let N : int>(VecN<N> a)
{
    return sqrt(lengthSquared(a));
}

VecN<N> normalize<let N : int>(VecN<N> a)
{
    float len = length(a);
    if (len > 0.0001)
        return a / len;
    return a;
}

// Generic NxN matrix
struct MatN<let N : int>
{
    float data[N][N]; // Row-major storage

    __init()
    {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            [ForceUnroll]
            for (int j = 0; j < N; j++)
                data[i][j] = 0.0;
    }

    static MatN<N> identity()
    {
        MatN<N> result;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i][i] = 1.0;
        return result;
    }

    float get(int row, int col) { return data[row][col]; }
    [mutating] void set(int row, int col, float val) { data[row][col] = val; }
}

// Matrix-vector multiplication
VecN<N> operator*<let N : int>(MatN<N> m, VecN<N> v)
{
    VecN<N> result;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        float sum = 0.0;
        [ForceUnroll]
        for (int j = 0; j < N; j++)
            sum += m.data[i][j] * v[j];
        result[i] = sum;
    }
    return result;
}

// Matrix-matrix multiplication
MatN<N> operator*<let N : int>(MatN<N> a, MatN<N> b)
{
    MatN<N> result;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0;
            [ForceUnroll]
            for (int k = 0; k < N; k++)
                sum += a.data[i][k] * b.data[k][j];
            result.data[i][j] = sum;
        }
    }
    return result;
}

// 4x4 matrix inverse using adjugate method
float4x4 inverse4x4(float4x4 m)
{
    float n11 = m[0][0], n12 = m[0][1], n13 = m[0][2], n14 = m[0][3];
    float n21 = m[1][0], n22 = m[1][1], n23 = m[1][2], n24 = m[1][3];
    float n31 = m[2][0], n32 = m[2][1], n33 = m[2][2], n34 = m[2][3];
    float n41 = m[3][0], n42 = m[3][1], n43 = m[3][2], n44 = m[3][3];

    float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    float idet = 1.0 / det;

    float4x4 result;
    result[0][0] = t11 * idet;
    result[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    result[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    result[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    result[1][0] = t12 * idet;
    result[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    result[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    result[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    result[2][0] = t13 * idet;
    result[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    result[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    result[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    result[3][0] = t14 * idet;
    result[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    result[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    result[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return result;
}

// 4D cross product (returns normal to 3 vectors in 4D space)
// This is the generalization of cross product to 4D
float4 cross4D(float4 a, float4 b, float4 c)
{
    // The 4D cross product uses a 4x4 determinant expansion
    // Result is perpendicular to all three input vectors
    float4 result;

    // Using cofactor expansion along first row of:
    // | e1  e2  e3  e4  |
    // | a.x a.y a.z a.w |
    // | b.x b.y b.z b.w |
    // | c.x c.y c.z c.w |

    result.x = determinant(float3x3(
        float3(a.y, a.z, a.w),
        float3(b.y, b.z, b.w),
        float3(c.y, c.z, c.w)
    ));

    result.y = -determinant(float3x3(
        float3(a.x, a.z, a.w),
        float3(b.x, b.z, b.w),
        float3(c.x, c.z, c.w)
    ));

    result.z = determinant(float3x3(
        float3(a.x, a.y, a.w),
        float3(b.x, b.y, b.w),
        float3(c.x, c.y, c.w)
    ));

    result.w = -determinant(float3x3(
        float3(a.x, a.y, a.z),
        float3(b.x, b.y, b.z),
        float3(c.x, c.y, c.z)
    ));

    return result;
}

// Compute normal to tetrahedron face in 4D
float4 getNormal4D(float4 v0, float4 v1, float4 v2, float4 v3)
{
    float4 edge1 = v1 - v0;
    float4 edge2 = v2 - v0;
    float4 edge3 = v3 - v0;
    return normalize(cross4D(edge1, edge2, edge3));
}
