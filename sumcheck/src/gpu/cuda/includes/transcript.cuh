#ifndef __TRANSCRIPT__
#define __TRANSCRIPT__
#include <stdint.h>
#include "./barretenberg/ecc/curves/bn254/fr.cuh"

class Transcript {
    private:
        uint8_t* start;
        uint8_t* cursor;
        uint8_t* end;
        fr state;
    public:
        void init_transcript(uint8_t* start, uint8_t* cursor, uint8_t* end);
        __device__ void write_field_element(fr fe);
        __device__ fr read_field_element();
}

uint8_t* fe_to_u8(fr fe);

#endif
