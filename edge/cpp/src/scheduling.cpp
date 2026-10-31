#include "soundbubble/realtime.hpp"

#include <iostream>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>
#include <avrt.h>
#else
#include <cerrno>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif

namespace soundbubble {

SchedulingResult try_set_realtime_fifo(int priority) {
  SchedulingResult r;
  r.requested_priority = priority;
  if (priority <= 0) {
    r.applied = false;
    r.message = "skipped (priority<=0)";
    return r;
  }
#if defined(_WIN32)
  if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
    r.applied = false;
    r.message = "SetPriorityClass failed";
    return r;
  }
  r.applied = true;
  r.requested_policy = "HIGH_PRIORITY_CLASS";
  r.message = "applied";
  return r;
#else
  sched_param param{};
  param.sched_priority = priority;
  if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
    r.applied = false;
    if (errno == EPERM) {
      r.message = "permission denied (run with sudo/chrt or grant CAP_SYS_NICE)";
    } else {
      r.message = std::string("failed (") + std::strerror(errno) + ")";
    }
    return r;
  }
  r.applied = true;
  r.message = "applied";
  return r;
#endif
}

void try_lock_memory() {
#if !defined(_WIN32)
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    std::cerr << "[warn] mlockall failed\n";
  }
#endif
}

void try_set_mmcss() {
#if defined(_WIN32)
  DWORD task = 0;
  HANDLE h = AvSetMmThreadCharacteristicsW(L"Pro Audio", &task);
  if (h) AvRevertMmThreadCharacteristics(h);
#endif
}

ScopedTimerResolution::ScopedTimerResolution(unsigned period_ms) : period_ms_(period_ms == 0 ? 1 : period_ms) {
#if defined(_WIN32)
  active_ = timeBeginPeriod(period_ms_) == TIMERR_NOERROR;
#else
  active_ = false;
#endif
}

ScopedTimerResolution::~ScopedTimerResolution() {
#if defined(_WIN32)
  if (active_) timeEndPeriod(period_ms_);
#else
  (void)period_ms_;
#endif
}

}  // namespace soundbubble
