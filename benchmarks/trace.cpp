#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <cxxabi.h>
#include <cstring>
#include <atomic>
#include <unordered_map>
static FILE *out = nullptr;
static std::mutex output_mx, correlation_mx;
static bool registered = false, subscribed = false;
static CUpti_SubscriberHandle subscriber;
static std::atomic<uint64_t> current_step{0}, dropped_total{0}, unmarked_total{0};
static std::atomic<unsigned> error_total{0};
static std::unordered_map<uint32_t, uint64_t> correlations;
static int check(CUptiResult r, const char *s) {
    if (r == CUPTI_SUCCESS)
        return 0;
    error_total.fetch_add(1);
    const char *e;
    cuptiGetResultString(r, &e);
    fprintf(stderr, "CUPTI %s: %s\n", s, e);
    return (int)r;
}
static void CUPTIAPI api_callback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId,
                                  const void *data) {
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API && domain != CUPTI_CB_DOMAIN_DRIVER_API)
        return;
    auto *d = (const CUpti_CallbackData *)data;
    if (d->callbackSite != CUPTI_API_ENTER || d->correlationId == 0)
        return;
    uint64_t step = current_step.load(std::memory_order_acquire);
    if (!step)
        return;
    std::lock_guard<std::mutex> lock(correlation_mx);
    auto it = correlations.find(d->correlationId);
    if (it != correlations.end() && it->second != step) {
        error_total.fetch_add(1);
        return;
    }
    correlations[d->correlationId] = step;
}
static uint64_t step_for(uint32_t id) {
    std::lock_guard<std::mutex> lock(correlation_mx);
    auto it = correlations.find(id);
    if (it == correlations.end()) {
        unmarked_total.fetch_add(1);
        return 0;
    }
    return it->second;
}
static void CUPTIAPI request(uint8_t **buffer, size_t *size, size_t *max) {
    *size = 16 * 1024 * 1024;
    *max = 0;
    *buffer = (uint8_t *)std::malloc(*size);
}
static void CUPTIAPI complete(CUcontext ctx, uint32_t stream, uint8_t *buffer, size_t,
                              size_t valid) {
    CUpti_Activity *rec = nullptr;
    CUptiResult status;
    while ((status = cuptiActivityGetNextRecord(buffer, valid, &rec)) == CUPTI_SUCCESS) {
        std::lock_guard<std::mutex> lock(output_mx);
        if (!out)
            continue;
        if (rec->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
            rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto *k = (CUpti_ActivityKernel9 *)rec;
            uint64_t step = step_for(k->correlationId);
            int err = 0;
            char *dem = abi::__cxa_demangle(k->name, nullptr, nullptr, &err);
            const char *name = err == 0 ? dem : k->name;
            fprintf(
                out,
                "kernel\t%llu\t%llu\t%u\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t%d\t%d\t%u\t0\t%llu\t%u\t%s\n",
                (unsigned long long)k->start, (unsigned long long)k->end, k->streamId, k->gridX,
                k->gridY, k->gridZ, k->blockX, k->blockY, k->blockZ, k->registersPerThread,
                k->staticSharedMemory, k->dynamicSharedMemory, k->localMemoryPerThread,
                (unsigned long long)step, k->correlationId, name);
            std::free(dem);
        } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto *k = (CUpti_ActivityMemcpy5 *)rec;
            uint64_t step = step_for(k->correlationId);
            fprintf(out,
                    "copy\t%llu\t%llu\t%u\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t%llu\t%llu\t%u\tcopy_kind_"
                    "%u\n",
                    (unsigned long long)k->start, (unsigned long long)k->end, k->streamId,
                    (unsigned long long)k->bytes, (unsigned long long)step, k->correlationId,
                    k->copyKind);
        } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            auto *k = (CUpti_ActivityMemset4 *)rec;
            uint64_t step = step_for(k->correlationId);
            fprintf(
                out,
                "memset\t%llu\t%llu\t%u\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t%llu\t%llu\t%u\tmemset\n",
                (unsigned long long)k->start, (unsigned long long)k->end, k->streamId,
                (unsigned long long)k->bytes, (unsigned long long)step, k->correlationId);
        }
    }
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED)
        check(status, "next record");
    size_t dropped = 0;
    check(cuptiActivityGetNumDroppedRecords(ctx, stream, &dropped), "dropped records");
    if (dropped) {
        dropped_total.fetch_add(dropped);
        fprintf(stderr, "CUPTI DROPPED %zu\n", dropped);
    }
    std::free(buffer);
}
extern "C" int trace_start(const char *path) {
    if (!path || !*path)
        return -1;
    current_step.store(0);
    dropped_total.store(0);
    error_total.store(0);
    unmarked_total.store(0);
    correlations.clear();
    if (!registered) {
        if (check(cuptiActivityRegisterCallbacks(request, complete), "callbacks"))
            return 1;
        registered = true;
    }
    if (!subscribed) {
        if (check(cuptiSubscribe(&subscriber, api_callback, nullptr), "subscribe"))
            return 1;
        subscribed = true;
    }
    out = std::fopen(path, "w");
    if (!out)
        return -2;
    fprintf(out, "kind\tstart_ns\tend_ns\tstream\tgx\tgy\tgz\tbx\tby\tbz\tregisters\tstatic_"
                 "shared\tdynamic_shared\tlocal_bytes\tbytes\tstep_id\tcorrelation_id\tname\n");
    if (check(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "driver callbacks"))
        return 2;
    if (check(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API), "runtime callbacks"))
        return 2;
    if (check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "enable kernel"))
        return 2;
    if (check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY), "enable memcpy"))
        return 2;
    return check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET), "enable memset");
}
extern "C" void trace_mark(uint64_t step) { current_step.store(step, std::memory_order_release); }
extern "C" int trace_stop() {
    current_step.store(0);
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "flush");
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "disable kernel");
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY), "disable memcpy");
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET), "disable memset");
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "flush");
    check(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "disable driver callbacks");
    check(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API),
          "disable runtime callbacks");
    std::lock_guard<std::mutex> lock(output_mx);
    if (out) {
        std::fclose(out);
        out = nullptr;
    }
    return dropped_total.load() || error_total.load() || unmarked_total.load() ? -3 : 0;
}
extern "C" uint64_t trace_timestamp() {
    uint64_t t = 0;
    check(cuptiGetTimestamp(&t), "timestamp");
    return t;
}
extern "C" uint64_t trace_dropped() { return dropped_total.load(); }
extern "C" uint64_t trace_unmarked() { return unmarked_total.load(); }
extern "C" unsigned trace_errors() { return error_total.load(); }
