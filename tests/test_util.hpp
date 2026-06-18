#pragma once
#include <cstdio>
#include <cmath>

inline int g_test_failures = 0;

#define CHECK(cond) \
  do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
    ++g_test_failures; } } while (0)

#define CHECK_NEAR(a, b, eps) \
  do { double _a = (a), _b = (b); \
    if (std::fabs(_a - _b) > (eps)) { \
      std::fprintf(stderr, "FAIL %s:%d: CHECK_NEAR(%s, %s): %.10g vs %.10g\n", \
                   __FILE__, __LINE__, #a, #b, _a, _b); \
      ++g_test_failures; } } while (0)

#define RUN(fn) do { std::printf("  - %s\n", #fn); fn(); } while (0)

inline int test_summary() {
  if (g_test_failures) {
    std::fprintf(stderr, "%d failure(s)\n", g_test_failures);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
