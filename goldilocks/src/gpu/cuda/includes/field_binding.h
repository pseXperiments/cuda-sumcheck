#ifndef __FIELD_BINDING_H__
#define __FIELD_BINDING_H__

#include <stdint.h>

struct FieldBinding {
    uint64_t data;
};

struct QuadraticExtFieldBinding {
    uint64_t data[2];
};

#endif
