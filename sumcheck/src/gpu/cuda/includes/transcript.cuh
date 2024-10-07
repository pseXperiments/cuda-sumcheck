#ifndef __TRANSCRIPT__
#define __TRANSCRIPT__
#include "barretenberg/ecc/curves/bn254/fr.cuh"

using namespace bb;

class Transcript {
    private:
        uint8_t* start;
        uint8_t* cursor;
    public:
        fr state;
        __device__ void init_transcript(uint8_t* start, uint8_t* cursor, fr* state);
        __device__ void write_field_element(fr fe);
        __device__ fr read_field_element();
        __device__ fr squeeze_challenge();
        __device__ int get_size();
};

__device__ uint8_t* fe_to_u8(fr fe);

#endif
