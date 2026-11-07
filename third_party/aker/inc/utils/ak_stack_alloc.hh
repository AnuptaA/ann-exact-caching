#pragma once

#include <cstddef>

/* This header provides a small helper to allocate a temporary buffer on the stack.
 *
 * Aker's conversion callbacks are synchronous and expect the destination buffer to
 * remain valid only for the duration of the call chain that uses the converted data.
 * The ANNSCache public APIs do not store the converted float pointers.
 */

#if defined(_MSC_VER)
    #include <malloc.h>
    #define AKER_STACK_ALLOCA(bytes) _alloca(bytes)
#else
    #include <alloca.h>
    #define AKER_STACK_ALLOCA(bytes) alloca(bytes)
#endif

namespace aker
{
    /**
     * @brief Stack-allocated float buffer.
     *
     * This helper allocates `length` floats on the caller's stack frame.
     * The returned pointer is valid only until the allocating function returns.
     */
    class StackFloatBuffer
    {
    public:
        explicit StackFloatBuffer(std::size_t length) noexcept
            : length_(length),
              data_(length_ == 0 ? nullptr
                                 : static_cast<float*>(AKER_STACK_ALLOCA(sizeof(float) * length_)))
        {
        }

        StackFloatBuffer(const StackFloatBuffer&) = delete;
        StackFloatBuffer& operator=(const StackFloatBuffer&) = delete;

        float* data() noexcept { return data_; }
        const float* data() const noexcept { return data_; }
        std::size_t size() const noexcept { return length_; }

    private:
        std::size_t length_;
        float* data_;
    };
}
