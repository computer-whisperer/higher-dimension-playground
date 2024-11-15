use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
pub struct TET_VERTEX {
    position: [f32; 4],
    tex_coord: [f32; 3],
    cell: u32
}

const V_NNNN: [f32; 4] = [-1.0, -1.0, -1.0, -1.0];
const V_PNNN: [f32; 4] = [1.0, -1.0, -1.0, -1.0];
const V_PPNN: [f32; 4] = [1.0, 1.0, -1.0, -1.0];
const V_NPNN: [f32; 4] = [-1.0, 1.0, -1.0, -1.0];
const V_NNPN: [f32; 4] = [-1.0, -1.0, 1.0, -1.0];
const V_PNPN: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
const V_PPPN: [f32; 4] = [1.0, 1.0, 1.0, -1.0];
const V_NPPN: [f32; 4] = [-1.0, 1.0, 1.0, -1.0];

const V_NNNP: [f32; 4] = [-1.0, -1.0, -1.0, 1.0];
const V_PNNP: [f32; 4] = [1.0, -1.0, -1.0, 1.0];
const V_PPNP: [f32; 4] = [1.0, 1.0, -1.0, 1.0];
const V_NPNP: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];
const V_NNPP: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];
const V_PNPP: [f32; 4] = [1.0, -1.0, 1.0, 1.0];
const V_PPPP: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const V_NPPP: [f32; 4] = [-1.0, 1.0, 1.0, 1.0];


pub const TRI_VERTICES: &[[f32; 4]] = &[

    // Cell 0 PXXX
    V_PPPP, V_PPNP, V_PPPN,// PPXX
    V_PPPN, V_PPNP, V_PPNN,

    V_PNNN, V_PNNP, V_PNPN,// PNXX
    V_PNPN, V_PNNP, V_PNPP,

    V_PPPP, V_PNPP, V_PPPN, // PXPX
    V_PPPN, V_PNPP, V_PNPN,

    V_PPNP, V_PNNP, V_PPNN, // PXNX
    V_PPNN, V_PNNP, V_PNNN,

    // PXXP

    // PXXN

    // Cell 1 NXXX
    V_NPPP, V_NPNP, V_NPPN,// NPXX
    V_NPPN, V_NPNP, V_NPNN,

    V_NNPP, V_NNNP, V_NNPN,// NNXX
    V_NNPN, V_NNNP, V_NNNN,

    V_NPPP, V_NNPP, V_NPPN,// NXPX
    V_NPPN, V_NNPP, V_NNPN,

    V_NPNP, V_NNNP, V_NPNN, // NXNX
    V_NPNN, V_NNNP, V_NNNN,

    // NXXP
    // NXXN

    // Cell 2 XPXX
    // PPXX
    // NPXX
    V_PPPP, V_NPPP, V_PPPN,// XPPX
    V_PPPN, V_NPPP, V_NPPN,

    V_PPNP, V_NPNP, V_PPNN,// XPNX
    V_PPNN, V_NPNP, V_NPNN,

    // XPXP
    // XPXN

    // Cell 3 XNXX
    // PNXX
    // NNXX
    V_PNPP, V_NNPP, V_PNPN, // XNPX
    V_PNPN, V_NNPP, V_NNPN,

    V_PNNP, V_NNNP, V_PNNN,// XNNX
    V_PNNN, V_NNNP, V_NNNN,
    // XNXP
    // XNXN

    // Cell 4 XXPX
    // PXPX
    // NXPX
    // XPPX
    // XNPX
    // XXPP
    // XXPN

    // Cell 5 XXNX
    // PXNX
    // NXNX
    // XPNX
    // XNNX
    // XXNP
    // XXNN

    // Cell 6 XXXP
    V_NNNP, V_PNNP, V_NPNP, //XXNP
    V_NPNP, V_PNNP, V_PPNP,

    V_PNPP, V_NNPP, V_PPPP, //XXPP
    V_PPPP, V_NNPP, V_NPPP,

    V_PNNP, V_PNPP, V_PPNP, //PXXP
    V_PPNP, V_PNPP, V_PPPP,

    V_NNPP, V_NNNP, V_NPPP, //NXXP
    V_NPPP, V_NNNP, V_NPNP,

    V_NPNP, V_PPNP, V_NPPP, //XPXP
    V_NPPP, V_PPNP, V_PPPP,

    V_NNPN, V_PNPN, V_NNNN, //XNXP
    V_NNNN, V_PNPN, V_PNNN,

    // Cell 7 XXXN
    V_NNNN, V_PNNN, V_NPNN, //XXNN
    V_NPNN, V_PNNN, V_PPNN,

    V_PNPN, V_NNPN, V_PPPN, //XXPN
    V_PPPN, V_NNPN, V_NPPN,

    V_PNNN, V_PNPN, V_PPNN, //PXXN
    V_PPNN, V_PNPN, V_PPPN,

    V_NNPN, V_NNNN, V_NPPN, //NXXN
    V_NPPN, V_NNNN, V_NPNN,

    V_NPNN, V_PPNN, V_NPPN, //XPXN
    V_NPPN, V_PPNN, V_PPPN,

    V_NNPN, V_PNPN, V_NNNN, //XNXN
    V_NNNN, V_PNPN, V_PNNN,

];

const V_PPP: [f32; 3] = [1.0, 1.0, 1.0];
const V_PPN: [f32; 3] = [1.0, 1.0, -1.0];
const V_PNP: [f32; 3] = [1.0, -1.0, 1.0];
const V_PNN: [f32; 3] = [1.0, -1.0, -1.0];
const V_NPP: [f32; 3] = [-1.0, 1.0, 1.0];
const V_NPN: [f32; 3] = [-1.0, 1.0, -1.0];
const V_NNP: [f32; 3] = [-1.0, -1.0, 1.0];
const V_NNN: [f32; 3] = [-1.0, -1.0, -1.0];

// Cell 0 PXXX
const V_C0_PPP: TET_VERTEX = TET_VERTEX{position: V_PPPP, tex_coord: V_PPP, cell: 0};
const V_C0_PPN: TET_VERTEX = TET_VERTEX{position: V_PPPN, tex_coord: V_PPN, cell: 0};
const V_C0_PNP: TET_VERTEX = TET_VERTEX{position: V_PPNP, tex_coord: V_PNP, cell: 0};
const V_C0_PNN: TET_VERTEX = TET_VERTEX{position: V_PPNN, tex_coord: V_PNN, cell: 0};
const V_C0_NPP: TET_VERTEX = TET_VERTEX{position: V_PNPP, tex_coord: V_NPP, cell: 0};
const V_C0_NPN: TET_VERTEX = TET_VERTEX{position: V_PNPN, tex_coord: V_NPN, cell: 0};
const V_C0_NNP: TET_VERTEX = TET_VERTEX{position: V_PNNP, tex_coord: V_NNP, cell: 0};
const V_C0_NNN: TET_VERTEX = TET_VERTEX{position: V_PNNN, tex_coord: V_NNN, cell: 0};

// Cell 1 NXXX
const V_C1_PPP: TET_VERTEX = TET_VERTEX{position: V_NPPP, tex_coord: V_PPP, cell: 1};
const V_C1_PPN: TET_VERTEX = TET_VERTEX{position: V_NPPN, tex_coord: V_PPN, cell: 1};
const V_C1_PNP: TET_VERTEX = TET_VERTEX{position: V_NPNP, tex_coord: V_PNP, cell: 1};
const V_C1_PNN: TET_VERTEX = TET_VERTEX{position: V_NPNN, tex_coord: V_PNN, cell: 1};
const V_C1_NPP: TET_VERTEX = TET_VERTEX{position: V_NNPP, tex_coord: V_NPP, cell: 1};
const V_C1_NPN: TET_VERTEX = TET_VERTEX{position: V_NNPN, tex_coord: V_NPN, cell: 1};
const V_C1_NNP: TET_VERTEX = TET_VERTEX{position: V_NNNP, tex_coord: V_NNP, cell: 1};
const V_C1_NNN: TET_VERTEX = TET_VERTEX{position: V_NNNN, tex_coord: V_NNN, cell: 1};

// Cell 2 XPXX
const V_C2_PPP: TET_VERTEX = TET_VERTEX{position: V_PPPP, tex_coord: V_PPP, cell: 2};
const V_C2_PPN: TET_VERTEX = TET_VERTEX{position: V_PPPN, tex_coord: V_PPN, cell: 2};
const V_C2_PNP: TET_VERTEX = TET_VERTEX{position: V_PPNP, tex_coord: V_PNP, cell: 2};
const V_C2_PNN: TET_VERTEX = TET_VERTEX{position: V_PPNN, tex_coord: V_PNN, cell: 2};
const V_C2_NPP: TET_VERTEX = TET_VERTEX{position: V_NPPP, tex_coord: V_NPP, cell: 2};
const V_C2_NPN: TET_VERTEX = TET_VERTEX{position: V_NPPN, tex_coord: V_NPN, cell: 2};
const V_C2_NNP: TET_VERTEX = TET_VERTEX{position: V_NPNP, tex_coord: V_NNP, cell: 2};
const V_C2_NNN: TET_VERTEX = TET_VERTEX{position: V_NPNN, tex_coord: V_NNN, cell: 2};

// Cell 3 XNXX
const V_C3_PPP: TET_VERTEX = TET_VERTEX{position: V_PNPP, tex_coord: V_PPP, cell: 3};
const V_C3_PPN: TET_VERTEX = TET_VERTEX{position: V_PNPN, tex_coord: V_PPN, cell: 3};
const V_C3_PNP: TET_VERTEX = TET_VERTEX{position: V_PNNP, tex_coord: V_PNP, cell: 3};
const V_C3_PNN: TET_VERTEX = TET_VERTEX{position: V_PNNN, tex_coord: V_PNN, cell: 3};
const V_C3_NPP: TET_VERTEX = TET_VERTEX{position: V_NNPP, tex_coord: V_NPP, cell: 3};
const V_C3_NPN: TET_VERTEX = TET_VERTEX{position: V_NNPN, tex_coord: V_NPN, cell: 3};
const V_C3_NNP: TET_VERTEX = TET_VERTEX{position: V_NNNP, tex_coord: V_NNP, cell: 3};
const V_C3_NNN: TET_VERTEX = TET_VERTEX{position: V_NNNN, tex_coord: V_NNN, cell: 3};

// Cell 4 XXPX
const V_C4_PPP: TET_VERTEX = TET_VERTEX{position: V_PPPP, tex_coord: V_PPP, cell: 4};
const V_C4_PPN: TET_VERTEX = TET_VERTEX{position: V_PPPN, tex_coord: V_PPN, cell: 4};
const V_C4_PNP: TET_VERTEX = TET_VERTEX{position: V_PNPP, tex_coord: V_PNP, cell: 4};
const V_C4_PNN: TET_VERTEX = TET_VERTEX{position: V_PNPN, tex_coord: V_PNN, cell: 4};
const V_C4_NPP: TET_VERTEX = TET_VERTEX{position: V_NPPP, tex_coord: V_NPP, cell: 4};
const V_C4_NPN: TET_VERTEX = TET_VERTEX{position: V_NPPN, tex_coord: V_NPN, cell: 4};
const V_C4_NNP: TET_VERTEX = TET_VERTEX{position: V_NNPP, tex_coord: V_NNP, cell: 4};
const V_C4_NNN: TET_VERTEX = TET_VERTEX{position: V_NNPN, tex_coord: V_NNN, cell: 4};

// Cell 5 XXNX
const V_C5_PPP: TET_VERTEX = TET_VERTEX{position: V_PPNP, tex_coord: V_PPP, cell: 5};
const V_C5_PPN: TET_VERTEX = TET_VERTEX{position: V_PPNN, tex_coord: V_PPN, cell: 5};
const V_C5_PNP: TET_VERTEX = TET_VERTEX{position: V_PNNP, tex_coord: V_PNP, cell: 5};
const V_C5_PNN: TET_VERTEX = TET_VERTEX{position: V_PNNN, tex_coord: V_PNN, cell: 5};
const V_C5_NPP: TET_VERTEX = TET_VERTEX{position: V_NPNP, tex_coord: V_NPP, cell: 5};
const V_C5_NPN: TET_VERTEX = TET_VERTEX{position: V_NPNN, tex_coord: V_NPN, cell: 5};
const V_C5_NNP: TET_VERTEX = TET_VERTEX{position: V_NNNP, tex_coord: V_NNP, cell: 5};
const V_C5_NNN: TET_VERTEX = TET_VERTEX{position: V_NNNN, tex_coord: V_NNN, cell: 5};

// Cell 6 XXXP
const V_C6_PPP: TET_VERTEX = TET_VERTEX{position: V_PPPP, tex_coord: V_PPP, cell: 6};
const V_C6_PPN: TET_VERTEX = TET_VERTEX{position: V_PPNP, tex_coord: V_PPN, cell: 6};
const V_C6_PNP: TET_VERTEX = TET_VERTEX{position: V_PNPP, tex_coord: V_PNP, cell: 6};
const V_C6_PNN: TET_VERTEX = TET_VERTEX{position: V_PNNP, tex_coord: V_PNN, cell: 6};
const V_C6_NPP: TET_VERTEX = TET_VERTEX{position: V_NPPP, tex_coord: V_NPP, cell: 6};
const V_C6_NPN: TET_VERTEX = TET_VERTEX{position: V_NPNP, tex_coord: V_NPN, cell: 6};
const V_C6_NNP: TET_VERTEX = TET_VERTEX{position: V_NNPP, tex_coord: V_NNP, cell: 6};
const V_C6_NNN: TET_VERTEX = TET_VERTEX{position: V_NNNP, tex_coord: V_NNN, cell: 6};

// Cell 7 XXXN
const V_C7_PPP: TET_VERTEX = TET_VERTEX{position: V_PPPN, tex_coord: V_PPP, cell: 7};
const V_C7_PPN: TET_VERTEX = TET_VERTEX{position: V_PPNN, tex_coord: V_PPN, cell: 7};
const V_C7_PNP: TET_VERTEX = TET_VERTEX{position: V_PNPN, tex_coord: V_PNP, cell: 7};
const V_C7_PNN: TET_VERTEX = TET_VERTEX{position: V_PNNN, tex_coord: V_PNN, cell: 7};
const V_C7_NPP: TET_VERTEX = TET_VERTEX{position: V_NPPN, tex_coord: V_NPP, cell: 7};
const V_C7_NPN: TET_VERTEX = TET_VERTEX{position: V_NPNN, tex_coord: V_NPN, cell: 7};
const V_C7_NNP: TET_VERTEX = TET_VERTEX{position: V_NNPN, tex_coord: V_NNP, cell: 7};
const V_C7_NNN: TET_VERTEX = TET_VERTEX{position: V_NNNN, tex_coord: V_NNN, cell: 7};


pub const TET_VERTICES: &[TET_VERTEX] = &[

    // Cell 0 PXXX
    V_C0_NNN, V_C0_NPN, V_C0_NNP, V_C0_PNN,
    V_C0_NPP, V_C0_NPN, V_C0_NNP, V_C0_PPP,
    V_C0_PNN, V_C0_NPN, V_C0_NNP, V_C0_PPP,
    V_C0_PNP, V_C0_PPP, V_C0_PNN, V_C0_NNP,
    V_C0_PPN, V_C0_PPP, V_C0_PNN, V_C0_NPN,
    // Cell 1 NXXX
    V_C1_NNN, V_C1_NPN, V_C1_NNP, V_C1_PNN,
    V_C1_NPP, V_C1_NPN, V_C1_NNP, V_C1_PPP,
    V_C1_PNN, V_C1_NPN, V_C1_NNP, V_C1_PPP,
    V_C1_PNP, V_C1_PPP, V_C1_PNN, V_C1_NNP,
    V_C1_PPN, V_C1_PPP, V_C1_PNN, V_C1_NPN,
    // Cell 2 XPXX
    V_C2_NNN, V_C2_NPN, V_C2_NNP, V_C2_PNN,
    V_C2_NPP, V_C2_NPN, V_C2_NNP, V_C2_PPP,
    V_C2_PNN, V_C2_NPN, V_C2_NNP, V_C2_PPP,
    V_C2_PNP, V_C2_PPP, V_C2_PNN, V_C2_NNP,
    V_C2_PPN, V_C2_PPP, V_C2_PNN, V_C2_NPN,
    // Cell 3 XNXX
    V_C3_NNN, V_C3_NPN, V_C3_NNP, V_C3_PNN,
    V_C3_NPP, V_C3_NPN, V_C3_NNP, V_C3_PPP,
    V_C3_PNN, V_C3_NPN, V_C3_NNP, V_C3_PPP,
    V_C3_PNP, V_C3_PPP, V_C3_PNN, V_C3_NNP,
    V_C3_PPN, V_C3_PPP, V_C3_PNN, V_C3_NPN,
    // Cell 4 XXPX
    V_C4_NNN, V_C4_NPN, V_C4_NNP, V_C4_PNN,
    V_C4_NPP, V_C4_NPN, V_C4_NNP, V_C4_PPP,
    V_C4_PNN, V_C4_NPN, V_C4_NNP, V_C4_PPP,
    V_C4_PNP, V_C4_PPP, V_C4_PNN, V_C4_NNP,
    V_C4_PPN, V_C4_PPP, V_C4_PNN, V_C4_NPN,
    // Cell 5 XXNX
    V_C5_NNN, V_C5_NPN, V_C5_NNP, V_C5_PNN,
    V_C5_NPP, V_C5_NPN, V_C5_NNP, V_C5_PPP,
    V_C5_PNN, V_C5_NPN, V_C5_NNP, V_C5_PPP,
    V_C5_PNP, V_C5_PPP, V_C5_PNN, V_C5_NNP,
    V_C5_PPN, V_C5_PPP, V_C5_PNN, V_C5_NPN,
    // Cell 6 XXXP
    V_C6_NNN, V_C6_NPN, V_C6_NNP, V_C6_PNN,
    V_C6_NPP, V_C6_NPN, V_C6_NNP, V_C6_PPP,
    V_C6_PNN, V_C6_NPN, V_C6_NNP, V_C6_PPP,
    V_C6_PNP, V_C6_PPP, V_C6_PNN, V_C6_NNP,
    V_C6_PPN, V_C6_PPP, V_C6_PNN, V_C6_NPN,
    // Cell 7 XXXN
    V_C7_NNN, V_C7_NPN, V_C7_NNP, V_C7_PNN,
    V_C7_NPP, V_C7_NPN, V_C7_NNP, V_C7_PPP,
    V_C7_PNN, V_C7_NPN, V_C7_NNP, V_C7_PPP,
    V_C7_PNP, V_C7_PPP, V_C7_PNN, V_C7_NNP,
    V_C7_PPN, V_C7_PPP, V_C7_PNN, V_C7_NPN,
];