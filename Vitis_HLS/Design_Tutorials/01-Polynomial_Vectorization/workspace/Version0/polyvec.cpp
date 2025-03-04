#include "polyvec.h" // Include the header file that contains the function declarations

// Define zetas array as constant
const int16_t zetas[K] = {
  -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
   -171,   622,  1577,   182,   962, -1202, -1474,  1468,
    573, -1325,   264,   383,  -829,  1458, -1602,  -130,
   -681,  1017,   732,   608, -1542,   411,  -205, -1571,
   1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
    516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
   -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
   -398,   961, -1508,  -725,   448, -1065,   677, -1275,
  -1103,   430,   555,   843, -1251,   871,  1550,   105,
    422,   587,   177,  -235,  -291,  -460,  1574,  1653,
   -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
  -1590,   644,  -872,   349,   418,   329,  -156,   -75,
    817,  1097,   603,   610,  1322, -1285, -1465,   384,
  -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
  -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
   -108,  -308,   996,   991,   958, -1460,  1522,  1628
};

void polyvec_ntt(polyvec *v) {
    polyvec_ntt_loop:for( unsigned int i = 0; i < K; ++i)
        poly_ntt(v->vec+i);
}
 
void poly_ntt(poly *a) {
	ntt(a->coeff);
}
 
 // Implementation of Montgomery reduction
int16_t montgomery_reduce(int32_t a) {
#pragma HLS INLINE
    int16_t t1, t2;
    t1 = (int16_t)a*QINV;
    t2 = (a - (int32_t)t1*Q) >> 16;
    return t2;
}

int16_t fqmul(int16_t a, int16_t b) {
#pragma HLS INLINE
    return montgomery_reduce((int32_t)a*b);
}

void ntt(uint16_t p[N]) {
    #pragma HLS INLINE OFF
    unsigned int len, start, j, k;
    int16_t t, zeta;

    k = 1;
    ntt_stage_outer:for(len = K; len >= 2; len >>= 1) {
        ntt_stage_loop:for(start = 0; start < 256; start = j + len) {
            zeta = zetas[k++];
            ntt_stage_inner:for(j = start; j < start + len; j++) {
                // #pragma HLS LOOP_TRIPCOUNT max=256+128
                #pragma HLS PIPELINE
                t = fqmul(zeta, p[j + len]);
                p[j + len] = p[j] - t;
                p[j] = p[j] + t;
            }
        }
    }
}
