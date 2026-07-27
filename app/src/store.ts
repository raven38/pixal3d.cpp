// Persistent gallery of past generations, backed by IndexedDB. GLBs are
// 10–15 MB each — far over the ~5 MB localStorage cap — so blobs live here.
//
// IndexedDB isn't always available: some WebKitGTK builds can't create their
// backing file ("Unable to establish IDB database file"), and private-mode /
// sandboxed webviews reject it outright. When that happens we must NOT lose a
// generation — the store transparently falls back to an in-memory map so the
// gallery still works for the session, and the caller's auto-save to the output
// folder (the real, on-disk deliverable) is unaffected.

import type { GenRecord } from "./types";

const DB_NAME = "trellis-studio";
const DB_VERSION = 1;
const STORE = "generations";

let dbp: Promise<IDBDatabase> | null = null;
let useMemory = false;
const mem = new Map<string, GenRecord>();

function db(): Promise<IDBDatabase> {
  if (dbp) return dbp;
  dbp = new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB unavailable"));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const d = req.result;
      if (!d.objectStoreNames.contains(STORE)) {
        const os = d.createObjectStore(STORE, { keyPath: "id" });
        os.createIndex("ts", "ts");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error("IndexedDB blocked"));
  });
  return dbp;
}

function tx(mode: IDBTransactionMode): Promise<IDBObjectStore> {
  return db().then((d) => d.transaction(STORE, mode).objectStore(STORE));
}

function wrap<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** Switch to the in-memory fallback and warn once. */
function fallback(e: unknown): void {
  if (!useMemory) {
    useMemory = true;
    console.warn(
      "IndexedDB unavailable — gallery persistence disabled for this session " +
        "(generations are still saved to the output folder).",
      e,
    );
  }
}

export function newId(): string {
  // Not crypto-sensitive; unique enough for gallery keys.
  return `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e9).toString(36)}`;
}

export async function put(rec: GenRecord): Promise<void> {
  if (useMemory) {
    mem.set(rec.id, rec);
    return;
  }
  try {
    await wrap((await tx("readwrite")).put(rec));
  } catch (e) {
    fallback(e);
    mem.set(rec.id, rec);
  }
}

export async function all(): Promise<GenRecord[]> {
  if (useMemory) {
    return [...mem.values()].sort((a, b) => b.ts - a.ts);
  }
  try {
    const recs = await wrap((await tx("readonly")).getAll() as IDBRequest<GenRecord[]>);
    return recs.sort((a, b) => b.ts - a.ts);
  } catch (e) {
    fallback(e);
    return [...mem.values()].sort((a, b) => b.ts - a.ts);
  }
}

export async function get(id: string): Promise<GenRecord | undefined> {
  if (useMemory) return mem.get(id);
  try {
    return await wrap((await tx("readonly")).get(id) as IDBRequest<GenRecord | undefined>);
  } catch (e) {
    fallback(e);
    return mem.get(id);
  }
}

export async function del(id: string): Promise<void> {
  if (useMemory) {
    mem.delete(id);
    return;
  }
  try {
    await wrap((await tx("readwrite")).delete(id));
  } catch (e) {
    fallback(e);
    mem.delete(id);
  }
}

export async function clear(): Promise<void> {
  mem.clear();
  if (useMemory) return;
  try {
    await wrap((await tx("readwrite")).clear());
  } catch (e) {
    fallback(e);
  }
}

/** True once the store has fallen back to in-memory (no cross-session persistence). */
export function isEphemeral(): boolean {
  return useMemory;
}
