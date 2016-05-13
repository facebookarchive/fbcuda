#pragma once

namespace facebook { namespace cuda {

void cudaSleep(THCState *state, int64_t cycles, int type);

}
