#include "./transcript.cuh"

void Transcript::init_transcript(uint8_t* s, uint8_t* c, uint8_t* e) {
    start = s;
    cursor = c;
    end = e;
}

__device__ fr Transcript::read_field_element() {
    if (c >= e) { return fr(0); }
    else {

    }
}

__device__ void Transcript::write_field_element(uint8_t* start, uint8_t* transcript, fr fe) {
    state = fe;
    uint8_t* new_address = transcript - 32;
    if (new_address < start) {
        return;
    }
    uint8_t* acc = new_address;
    for ()
}

uint8_t* fe_to_u8(fr fe) {
    
}
