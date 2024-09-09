#include "transcript.cuh"

__device__ void Transcript::init_transcript(uint8_t* s, uint8_t* c, uint8_t* e) {
    start = s;
    cursor = c;
    end = e;
    state = fr(0);
}

__device__ fr Transcript::read_field_element() {
    if (cursor > end - 32) { return fr(0); }
    else {
        uint64_t data[4] = {0, 0, 0, 0};
        for (int i = 0; i < 32; i++) {
            data[i % 4] += cursor[i] * ((i % 4) * 256);
        }
        fr fe = fr(data[0], data[1], data[2], data[3]);
        cursor += 32;
        state = fe;
        return fe;
    }
}

__device__ void Transcript::write_field_element(fr fe) {
    state = fe;
    cursor -= 32;
    if (cursor < start) {
        cursor += 32;
        return;
    }
    uint8_t* write_fe = fe_to_u8(fe);
    for (int i = 0; i < 32; i++) {
        *(cursor + i) = write_fe[i];
    }
    return;
}

__device__ fr Transcript::squeeze_challenge() {
    return fr(1034);
}

__device__ uint8_t* fe_to_u8(fr fe) {
    fe.self_to_montgomery_form();
    uint8_t result[32];
    for (int i = 0; i < 4; i++) {
        memcpy(&result[i * 8], &fe.data[i], sizeof(fe.data[i]));
    }
    return result;
}
