
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


pub const VERTICES: &[[f32; 4]] = &[

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