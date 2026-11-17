// Link stubs for the standalone ProcessKV_v2 repro (no TM libs linked).
#include <cstdio>
#include <cuda_runtime.h>
#include <sstream>
#include <string>

namespace turbomind {

void ReportCudaError(cudaError_t ec, const char* file, int line)
{
    fprintf(stderr, "stub cuda error %d at %s:%d\n", (int)ec, file, line);
}

}  // namespace turbomind

namespace turbomind {
namespace core {

class CheckOpStringBuilder {
public:
    CheckOpStringBuilder();
    std::ostream* ForVal1();
    std::ostream* ForVal2();
    std::string*  NewString();
};

CheckOpStringBuilder::CheckOpStringBuilder() {}
std::ostream* CheckOpStringBuilder::ForVal1()
{
    static std::ostringstream s;
    return &s;
}
std::ostream* CheckOpStringBuilder::ForVal2()
{
    static std::ostringstream s;
    return &s;
}
std::string* CheckOpStringBuilder::NewString()
{
    return new std::string("chk");
}

class CheckErrorStream {
public:
    CheckErrorStream(const char*, int, const char*);
    CheckErrorStream(const char*, int, const char*, std::string*);
    void Report();
};

CheckErrorStream::CheckErrorStream(const char*, int, const char*) {}
CheckErrorStream::CheckErrorStream(const char*, int, const char*, std::string*) {}
void         CheckErrorStream::Report() {}

}  // namespace core
}  // namespace turbomind
