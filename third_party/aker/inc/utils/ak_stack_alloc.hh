#pragma once

#include <cstddef>
#include <vector>

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
     * @brief Float buffer for one converted query vector.
     *
     * Local patch 5: upstream allocated with alloca() inside this constructor, so the
     * buffer was released as soon as the constructor returned and every converted
     * query read freed stack memory (observed: garbage vectors, approximate lookups
     * never hit, with clang on macOS). The storage is now owned by the object.
     */
    class StackFloatBuffer
    {
    public:
        explicit StackFloatBuffer(std::size_t length) noexcept
            : length_(length), storage_(length)
        {
        }

        StackFloatBuffer(const StackFloatBuffer&) = delete;
        StackFloatBuffer& operator=(const StackFloatBuffer&) = delete;

        float* data() noexcept { return storage_.data(); }
        const float* data() const noexcept { return storage_.data(); }
        std::size_t size() const noexcept { return length_; }

    private:
        std::size_t length_;
        std::vector<float> storage_;
    };
}
