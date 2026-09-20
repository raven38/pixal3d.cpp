// Generate ボタンを押してよいかを 3 つのモード（TRELLIS.2 / Pixal3D SV / Pixal3D MV）で
// 共有する単一のゲート。
//
// trellis-server は生成を 1 本ずつ直列に走らせ、クライアントが応答待ちをやめても
// （AbortController）サーバ側の計算は止まらない。だから「Stop waiting」の後にすぐ次を
// 押させると、前の生成が終わるまで mutex 待ちになる。それを防ぐために、サーバが
// GET /capabilities で申告する busy / completed を見て、
//   - busy=true の間は全モードで Generate を無効
//   - Stop waiting を押した時点の completed を覚え、busy=false かつ completed が増えたら
//     「前の生成は終わった」と判断して Generate を戻す
// という規則にする（設計書 D11）。/capabilities を持たない 0.9.0 のサーバでは busy 情報が
// 無いので、従来どおり待機中止 = 即再有効化にする。

import type { Capabilities } from "./types";

export interface GateState {
  serverOnline: boolean;
  /** このウィンドウが応答を待っている生成があるか（モードを問わない）。 */
  localGenerating: boolean;
  /** サーバが申告した busy。/capabilities が無い旧サーバでは常に false。 */
  serverBusy: boolean;
  /** Stop waiting の後、サーバ側の完了をまだ確認できていない。 */
  awaitingServer: boolean;
  /** 直近の /capabilities。null は未取得か旧サーバ。 */
  capabilities: Capabilities | null;
}

const state: GateState = {
  serverOnline: false,
  localGenerating: false,
  serverBusy: false,
  awaitingServer: false,
  capabilities: null,
};

let completedAtStop = -1;
const listeners = new Set<(s: GateState) => void>();
const lostListeners = new Set<(msg: string) => void>();

function emit(): void {
  for (const fn of listeners) fn(state);
}

export function gateState(): Readonly<GateState> {
  return state;
}

export function subscribeGate(fn: (s: GateState) => void): () => void {
  listeners.add(fn);
  fn(state);
  return () => listeners.delete(fn);
}

/** 生成を開始できる条件。入力の有無はモードごとに別途見る。 */
export function canGenerate(): boolean {
  return state.serverOnline && !state.localGenerating && !state.serverBusy && !state.awaitingServer;
}

/** 生成が待ちに入ったときに Generate が無効な理由（表示用）。null なら押せる。 */
export function blockedReason(): string | null {
  if (!state.serverOnline) return "server is offline";
  if (state.localGenerating) return "a generation is running";
  if (state.awaitingServer) return "the server is still finishing the generation you stopped waiting for";
  if (state.serverBusy) return "the server is busy with another generation";
  return null;
}

export function setServerOnline(online: boolean): void {
  if (state.serverOnline === online) return;
  state.serverOnline = online;
  emit();
}

export function setLocalGenerating(on: boolean): void {
  if (state.localGenerating === on) return;
  state.localGenerating = on;
  emit();
}

/**
 * Stop waiting: 応答待ちをやめた。サーバはまだ計算しているので、その完了を
 * /capabilities で確認できるまで Generate を止める。旧サーバ（capabilities=null）では
 * 確認手段が無いので止めない（0.9.0 と同じ挙動）。
 */
export function stopWaiting(): void {
  if (!state.capabilities) {
    state.awaitingServer = false;
  } else {
    completedAtStop = state.capabilities.completed;
    state.awaitingServer = true;
  }
  emit();
}

/**
 * サーバ消失（#36）: Tauri の `server-exited` か、生成中に /health が続けて落ちたとき。
 * 応答待ちのパネルはこれで fetch を abort し、進行中の状態を畳む（サーバは死んでいるので
 * Stop waiting と違って「まだ計算中」ではない）。ゲートは offline に戻す。
 */
export function onServerLost(fn: (msg: string) => void): () => void {
  lostListeners.add(fn);
  return () => lostListeners.delete(fn);
}

export function notifyServerLost(msg: string): void {
  state.serverOnline = false;
  state.serverBusy = false;
  state.awaitingServer = false;
  state.capabilities = null;
  for (const fn of lostListeners) fn(msg);
  emit();
}

/** ポーリングごとに呼ぶ。null は「取れなかった / 旧サーバ」。 */
export function observeCapabilities(cap: Capabilities | null): void {
  state.capabilities = cap;
  if (!cap) {
    // 旧サーバや一時的な取得失敗。busy は分からないので押せる側に倒し、
    // awaiting は解除する（永久に押せなくなるのを避ける）。
    state.serverBusy = false;
    state.awaitingServer = false;
    emit();
    return;
  }
  state.serverBusy = cap.busy;
  if (state.awaitingServer && !cap.busy && cap.completed > completedAtStop) {
    state.awaitingServer = false;
  }
  emit();
}
