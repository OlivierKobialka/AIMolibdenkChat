const ENV_API_BASE_URL = process.env.NEXT_PUBLIC_API_URL?.trim();

function resolveApiBaseUrl(): string {
	if (ENV_API_BASE_URL) {
		return ENV_API_BASE_URL.replace(/\/+$/, "");
	}

	if (typeof window !== "undefined") {
		const host = window.location.hostname;
		if (host === "localhost" || host === "127.0.0.1") {
			return "http://localhost:5000";
		}
	}

	// Vercel experimentalServices backend route prefix (same-origin).
	return "/_/backend";
}

const API_BASE_URL = resolveApiBaseUrl();

export interface LoginRequest {
	email: string;
	password: string;
}

export interface LoginResponse {
	success: boolean;
	session_id: string;
	message: string;
}

export interface ChatRequest {
	session_id: string;
	message: string;
}

export interface ChatResponse {
	response: string;
	sources: string[];
}

type ErrorResponse = {
	error: string;
};

async function parseJsonSafe<T>(response: Response): Promise<T | null> {
	try {
		return (await response.json()) as T;
	} catch {
		return null;
	}
}

export async function loginRequest(
	payload: LoginRequest,
): Promise<LoginResponse> {
	const response = await fetch(`${API_BASE_URL}/api/login`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(payload),
	});

	const data = await parseJsonSafe<LoginResponse | ErrorResponse>(response);

	if (!response.ok) {
		throw new Error(
			(data as ErrorResponse | null)?.error ||
				"Logowanie nie powiodło się.",
		);
	}

	return data as LoginResponse;
}

export async function chatRequest(payload: ChatRequest): Promise<ChatResponse> {
	const response = await fetch(`${API_BASE_URL}/api/chat`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(payload),
	});

	const data = await parseJsonSafe<ChatResponse | ErrorResponse>(response);

	if (!response.ok) {
		throw new Error(
			(data as ErrorResponse | null)?.error || "Błąd komunikacji z AI.",
		);
	}

	return data as ChatResponse;
}
