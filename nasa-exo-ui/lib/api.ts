const BASE_URL = "http://127.0.0.1:7860/api";

export async function getMetrics() {
  return fetch(`${BASE_URL}/metrics`).then(res => res.json());
}

export async function predict(payload: any) {
  return fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(res => res.json());
}

export async function getCandidates() {
  return fetch(`${BASE_URL}/candidates?topK=50`).then(res => res.json());
}

export async function getExplanation(starId: string) {
  return fetch(`${BASE_URL}/explanations?star_id=${starId}`).then(res => res.json());
}
