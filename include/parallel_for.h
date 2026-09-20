// 均等分割の parallel_for（std::thread、実行時ライブラリ不要）。src/remesh_dc.cpp の同名関数と
// 同じ流儀で、Emscripten ビルド（pthread 無しでリンク、std::thread の生成が
// "thread constructor failed: Not supported" で throw する）では直列に落とす。
// fn(begin, end) は [begin, end) の範囲を処理する。呼び出し側は fn が書く先が範囲ごとに
// 互いに素であること（または順序非依存の atomic 操作であること）を保証する。
#pragma once
#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace trellis {

inline int parallel_threads() {
#ifdef __EMSCRIPTEN__
    return 1;
#else
    return (int)std::max(1u, std::thread::hardware_concurrency());
#endif
}

// nt: 使うスレッド数（既定は parallel_threads()）。分割は [b,e) の均等 chunk で、結果が
// 分割に依存しない処理にだけ使う。
template <class Fn>
void parallel_for(int64_t n, Fn&& fn, int nt = parallel_threads()) {
#ifdef __EMSCRIPTEN__
    nt = 1;   // 明示された nt も無視する（std::thread が生成できない）
#endif
    if (nt <= 1 || n < 4096) { if (n > 0) fn((int64_t)0, n); return; }   // 小さい仕事は直列
    std::vector<std::thread> ts;
    ts.reserve((size_t)nt);
    const int64_t chunk = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int64_t b = (int64_t)t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        ts.emplace_back([&fn, b, e] { fn(b, e); });
    }
    for (auto& t : ts) t.join();
}

}  // namespace trellis
