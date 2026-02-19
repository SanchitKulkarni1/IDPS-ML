// // src/lib/api.ts
// //const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:5000";

// const API_BASE = "http://localhost:5000";

// async function jsonPost(path: string, body: any) {
//   const res = await fetch(`${API_BASE}${path}`, {
//     method: "POST",
//     headers: { "Content-Type": "application/json" },
//     body: JSON.stringify(body),
//   });
//   if (!res.ok) throw await res.json();
//   return res.json();
// }

// async function filePost(path: string, formData: FormData) {
//   const res = await fetch(`${API_BASE}${path}`, {
//     method: "POST",
//     body: formData,
//   });
//   if (!res.ok) throw await res.json();
//   return res.json();
// }

// async function getJson(path: string) {
//   const res = await fetch(`${API_BASE}${path}`);
//   if (!res.ok) throw await res.json();
//   return res.json();
// }

// export { API_BASE, jsonPost, filePost, getJson };



// src/lib/api.ts
const API_BASE = "http://localhost:5000";

function buildUrl(path: string) {
  if (!path) return API_BASE;
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE.replace(/\/+$/, "")}${p}`;
}

async function parseResponse(res: Response) {
  const txt = await res.text();
  try {
    return txt ? JSON.parse(txt) : null;
  } catch {
    return txt;
  }
}

async function jsonPost(path: string, body: any, extraHeaders: Record<string, string> = {}) {
  const url = buildUrl(path);
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: JSON.stringify(body),
  });
  const parsed = await parseResponse(res);
  if (!res.ok) {
    const err = (parsed && typeof parsed === "object") ? parsed : { error: parsed || `${res.status} ${res.statusText}` };
    throw err;
  }
  return parsed;
}

async function filePost(path: string, formData: FormData, extraHeaders: Record<string, string> = {}) {
  const url = buildUrl(path);
  const res = await fetch(url, {
    method: "POST",
    headers: { ...extraHeaders }, // do NOT set Content-Type here
    body: formData,
  });
  const parsed = await parseResponse(res);
  if (!res.ok) {
    const err = (parsed && typeof parsed === "object") ? parsed : { error: parsed || `${res.status} ${res.statusText}` };
    throw err;
  }
  return parsed;
}

async function getJson(path: string, params?: Record<string, string | number | boolean>) {
  const base = buildUrl(path);
  let url = base;
  if (params && Object.keys(params).length) {
    const qs = new URLSearchParams();
    for (const k of Object.keys(params)) qs.append(k, String(params[k]));
    url = `${base}${base.includes("?") ? "&" : "?"}${qs.toString()}`;
  }
  const res = await fetch(url);
  const parsed = await parseResponse(res);
  if (!res.ok) {
    const err = (parsed && typeof parsed === "object") ? parsed : { error: parsed || `${res.status} ${res.statusText}` };
    throw err;
  }
  return parsed;
}

async function putJson(path: string, body: any, extraHeaders: Record<string, string> = {}) {
  const url = buildUrl(path);
  const res = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: JSON.stringify(body),
  });
  const parsed = await parseResponse(res);
  if (!res.ok) {
    const err = (parsed && typeof parsed === "object") ? parsed : { error: parsed || `${res.status} ${res.statusText}` };
    throw err;
  }
  return parsed;
}

export { API_BASE, jsonPost, filePost, getJson, putJson };
export default { API_BASE, jsonPost, filePost, getJson, putJson };
