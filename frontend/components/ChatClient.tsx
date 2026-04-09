"use client";

import {
	useEffect,
	useMemo,
	useRef,
	useState,
	type FormEvent,
	type KeyboardEvent,
} from "react";
import Script from "next/script";
import { useRouter } from "next/navigation";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { chatRequest } from "../lib/api";
import { clearSession, getSession } from "../lib/session";

type MessageRole = "user" | "assistant" | "error";

type ChatMessage = {
	id: string;
	role: MessageRole;
	content: string;
	time: string;
	sources?: string[];
};

const SUGGESTED_QUESTIONS: string[] = [
	"Czym jest Moly i jaka jest jego glowna oferta?",
	"Jaki jest model finansowy Moly?",
	"Jakie sa kluczowe przewagi strategiczne Moly?",
	"Kim sa partnerzy B2B Moly?",
];

function getTime(): string {
	return new Date().toLocaleTimeString("pl-PL", {
		hour: "2-digit",
		minute: "2-digit",
	});
}

type MessageBubbleProps = {
	message: ChatMessage;
};

function MessageBubble({ message }: MessageBubbleProps) {
	const isUser = message.role === "user";
	const isError = message.role === "error";

	return (
		<article
			className='fade-in'
			style={{
				display: "flex",
				justifyContent: isUser ? "flex-end" : "flex-start",
			}}>
			<div
				style={{
					maxWidth: "82%",
					borderRadius: "9px",
					padding: "12px 14px",
					border: isUser
						? "1px solid #1d4ed8"
						: isError
							? "1px solid #fecaca"
							: "1px solid var(--line)",
					background: isUser
						? "#2563eb"
						: isError
							? "#fef2f2"
							: "#ffffff",
					color: isUser ? "#ffffff" : isError ? "#b91c1c" : "#0f172a",
				}}>
				{message.role === "assistant" ? (
					<div className='markdown-content'>
						<Markdown remarkPlugins={[remarkGfm]}>
							{message.content}
						</Markdown>
					</div>
				) : (
					<p
						style={{
							margin: 0,
							whiteSpace: "pre-wrap",
							lineHeight: 1.45,
						}}>
						{message.content}
					</p>
				)}
				{message.sources?.length ? (
					<div
						style={{
							marginTop: "9px",
							display: "flex",
							gap: "6px",
							flexWrap: "wrap",
						}}>
						{message.sources.map(source => (
							<span
								key={`${message.id}-${source}`}
								className='badge'
								style={{
									fontSize: "10px",
									background: "#f8fafc",
									color: "#1e3a8a",
								}}>
								{source}
							</span>
						))}
					</div>
				) : null}
				<div
					style={{
						marginTop: "8px",
						fontSize: "11px",
						opacity: 0.75,
					}}>
					{message.time}
				</div>
			</div>
		</article>
	);
}

export default function ChatClient() {
	const router = useRouter();
	const inputRef = useRef<HTMLTextAreaElement | null>(null);
	const endRef = useRef<HTMLDivElement | null>(null);

	const [ready, setReady] = useState<boolean>(false);
	const [sessionId, setSessionId] = useState<string>("");
	const [email, setEmail] = useState<string>("");
	const [text, setText] = useState<string>("");
	const [loading, setLoading] = useState<boolean>(false);
	const [messages, setMessages] = useState<ChatMessage[]>([]);

	useEffect(() => {
		const session = getSession();
		if (!session.sessionId) {
			router.replace("/");
			return;
		}

		setSessionId(session.sessionId);
		setEmail(session.email || session.sessionId);
		setReady(true);
	}, [router]);

	useEffect(() => {
		if (!endRef.current) {
			return;
		}

		endRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
	}, [messages, loading]);

	// Remove Spline Logo watermark dynamically
	useEffect(() => {
		const hideLogo = () => {
			const viewer = document.querySelector("spline-viewer");
			if (viewer && viewer.shadowRoot) {
				const logo = viewer.shadowRoot.querySelector("#logo");
				if (logo) {
					logo.remove();
				}
			}
		};

		const intervalId = setInterval(hideLogo, 100);
		setTimeout(() => clearInterval(intervalId), 8000);

		return () => clearInterval(intervalId);
	}, []);

	const canSend = useMemo(
		() => text.trim().length > 0 && !loading,
		[text, loading],
	);

	function logout(): void {
		clearSession();
		router.push("/");
	}

	function applySuggestion(question: string): void {
		if (loading) {
			return;
		}

		setText(question);
		inputRef.current?.focus();
	}

	async function sendMessage(explicitText?: string): Promise<void> {
		const clean = (explicitText ?? text).trim();
		if (!clean || loading) {
			return;
		}

		const userMsg: ChatMessage = {
			id: `${Date.now()}-u`,
			role: "user",
			content: clean,
			time: getTime(),
		};

		setLoading(true);
		setMessages(prev => [...prev, userMsg]);
		setText("");

		try {
			const data = await chatRequest({
				message: clean,
				session_id: sessionId,
			});

			setMessages(prev => [
				...prev,
				{
					id: `${Date.now()}-a`,
					role: "assistant",
					content: data?.response || "Brak odpowiedzi.",
					sources: Array.isArray(data?.sources) ? data.sources : [],
					time: getTime(),
				},
			]);
		} catch (err: unknown) {
			const errorMessage =
				err instanceof Error
					? err.message ||
						"Wystapil blad podczas polaczenia z serwerem."
					: "Wystapil blad podczas polaczenia z serwerem.";

			setMessages(prev => [
				...prev,
				{
					id: `${Date.now()}-e`,
					role: "error",
					content: errorMessage,
					time: getTime(),
				},
			]);
		} finally {
			setLoading(false);
		}
	}

	function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
		if (event.key === "Enter" && !event.shiftKey) {
			event.preventDefault();
			void sendMessage();
		}
	}

	if (!ready) {
		return null;
	}

	const hasMessages = messages.length > 0;

	return (
		<main
			className='page'
			style={{
				display: "flex",
				flexDirection: "column",
				minHeight: "100vh",
			}}>
			<Script
				type='module'
				src='https://unpkg.com/@splinetool/viewer@1.12.77/build/spline-viewer.js'
				strategy='afterInteractive'
			/>
			<header
				style={{
					borderBottom: "1px solid var(--line)",
					background: "rgba(255,255,255,0.85)",
					backdropFilter: "blur(6px)",
					position: "sticky",
					top: 0,
					zIndex: 10,
				}}>
				<div
					style={{
						maxWidth: "1050px",
						margin: "0 auto",
						padding: "12px 18px",
						display: "flex",
						justifyContent: "space-between",
						alignItems: "center",
						gap: "12px",
					}}>
					<div>
						<div
							className='badge'
							style={{
								marginBottom: "4px",
								display: "inline-block",
							}}>
							Moly AI Chat
						</div>
						<h1 style={{ margin: 0, fontSize: "16px" }}>
							Asystent biznesowy
						</h1>
					</div>

					<div
						style={{
							display: "flex",
							alignItems: "center",
							gap: "8px",
						}}>
						<span
							className='badge'
							style={{ background: "#ffffff" }}>
							{email}
						</span>
						<button
							className='button-outline'
							onClick={logout}
							type='button'>
							Wyloguj
						</button>
					</div>
				</div>
			</header>

			{!hasMessages ? (
				<div 
					className="empty-state-grid"
					style={{ 
						display: "grid", 
						gridTemplateColumns: "1.2fr 1fr", 
						flex: 1, 
						maxWidth: "1150px", 
						margin: "0 auto", 
						width: "100%", 
						alignItems: "center" 
					}}>
					<section
						style={{
							display: "flex",
							alignItems: "center",
							justifyContent: "center",
							padding: "20px 18px",
							width: "100%",
						}}>
						<div
							style={{
								width: "100%",
								maxWidth: "600px",
								display: "flex",
								flexDirection: "column",
								gap: "20px",
							}}>
						<div style={{ textAlign: "center" }}>
							<h2
								style={{
									marginTop: 0,
									marginBottom: "8px",
									fontSize: "24px",
									fontWeight: 600,
								}}>
								Witaj w Moly AI
							</h2>
							<p
								style={{
									marginTop: 0,
									marginBottom: 0,
									color: "var(--muted)",
									fontSize: "14px",
								}}>
								Zadawaj pytania o oferte, strategię i model
								biznesowy Moly.
							</p>
						</div>

						<form
							onSubmit={(event: FormEvent<HTMLFormElement>) => {
								event.preventDefault();
								void sendMessage();
							}}
							style={{
								display: "flex",
								gap: "10px",
								alignItems: "end",
							}}>
							<textarea
								ref={inputRef}
								className='input'
								value={text}
								onChange={event => setText(event.target.value)}
								onKeyDown={onKeyDown}
								placeholder='Zadaj pytanie o Moly...'
								rows={2}
								style={{
									resize: "none",
									minHeight: "50px",
									maxHeight: "150px",
									flex: 1,
								}}
							/>
							<button
								className='button-primary'
								type='submit'
								disabled={!canSend}
								style={{ minHeight: "50px" }}>
								{loading ? "Wysyłanie..." : "Wyślij"}
							</button>
						</form>

						<div
							style={{
								display: "grid",
								gap: "8px",
								gridTemplateColumns: "1fr 1fr",
							}}>
							{SUGGESTED_QUESTIONS.map(question => (
								<button
									type='button'
									key={question}
									className='button-outline'
									onClick={() => applySuggestion(question)}
									style={{
										textAlign: "left",
										minHeight: "48px",
										fontSize: "13px",
									}}>
									{question}
								</button>
							))}
						</div>
					</div>
					</section>
					<div className="spline-container" style={{ height: "550px", position: "relative" }}>
						<spline-viewer url="https://prod.spline.design/PEeua81IXcs20GAN/scene.splinecode"></spline-viewer>
					</div>
				</div>
			) : (
				<section
					style={{
						flex: 1,
						width: "100%",
						maxWidth: "1050px",
						margin: "0 auto",
						padding: "20px 18px 140px",
						display: "flex",
						flexDirection: "column",
					}}>
					<div style={{ display: "grid", gap: "12px" }}>
						{messages.map(message => (
							<MessageBubble key={message.id} message={message} />
						))}

						{loading ? (
							<div
								style={{
									display: "flex",
									justifyContent: "flex-start",
								}}>
								<div
									className='badge'
									style={{
										background: "#ffffff",
										color: "#1e40af",
									}}>
									Moly AI pisze...
								</div>
							</div>
						) : null}
						<div ref={endRef} />
					</div>
				</section>
			)}

			{hasMessages && (
				<footer
					style={{
						position: "fixed",
						bottom: 0,
						width: "100%",
						borderTop: "1px solid var(--line)",
						background: "rgba(255,255,255,0.96)",
						backdropFilter: "blur(7px)",
						zIndex: 5,
					}}>
					<form
						onSubmit={(event: FormEvent<HTMLFormElement>) => {
							event.preventDefault();
							void sendMessage();
						}}
						style={{
							maxWidth: "1050px",
							margin: "0 auto",
							padding: "12px 18px",
							display: "flex",
							gap: "10px",
							alignItems: "end",
						}}>
						<textarea
							ref={inputRef}
							className='input'
							value={text}
							onChange={event => setText(event.target.value)}
							onKeyDown={onKeyDown}
							placeholder='Zadaj pytanie o Moly...'
							rows={1}
							style={{
								resize: "none",
								minHeight: "44px",
								maxHeight: "150px",
								flex: 1,
							}}
						/>
						<button
							className='button-primary'
							type='submit'
							disabled={!canSend}>
							{loading ? "Wysyłanie..." : "Wyślij"}
						</button>
					</form>
				</footer>
			)}

			<style jsx>{`
				@media (max-width: 800px) {
					section div[style*="grid-template-columns: 1fr 1fr"] {
						grid-template-columns: 1fr !important;
					}

					.empty-state-grid {
						grid-template-columns: 1fr !important;
					}

					.spline-container {
						display: none;
					}

					form {
						flex-direction: column;
					}

					form button {
						width: 100%;
					}
				}
			`}</style>
		</main>
	);
}
