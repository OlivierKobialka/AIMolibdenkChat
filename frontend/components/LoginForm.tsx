"use client";

import { useMemo, useState, type ChangeEvent, type FormEvent } from "react";
import Script from "next/script";
import { useRouter } from "next/navigation";
import { loginRequest } from "../lib/api";
import { saveSession } from "../lib/session";

export default function LoginForm() {
	const [email, setEmail] = useState<string>("");
	const [password, setPassword] = useState<string>("");
	const [showPassword, setShowPassword] = useState<boolean>(false);
	const [loading, setLoading] = useState<boolean>(false);
	const [error, setError] = useState<string>("");
	const splineSceneUrl =
		"https://unpkg.com/@splinetool/viewer@1.12.74/build/spline-viewer.js" ||
		"undefined";

	const router = useRouter();
	const formDisabled = loading || !email.trim() || !password;

	const accentLabel = useMemo(() => {
		if (loading) {
			return "Logowanie";
		}
		return "Zaloguj";
	}, [loading]);

	async function onSubmit(event: FormEvent<HTMLFormElement>) {
		event.preventDefault();
		setError("");

		if (!email.trim() || !password) {
			setError("Wypełnij wszystkie pola.");
			return;
		}

		setLoading(true);

		try {
			const data = await loginRequest({
				email: email.trim(),
				password,
			});

			saveSession(data.session_id, email.trim());
			router.push("/chat");
		} catch (err: unknown) {
			if (err instanceof Error) {
				setError(err.message || "Nie udało się zalogować.");
			} else {
				setError("Nie udało się zalogować.");
			}
		} finally {
			setLoading(false);
		}
	}

	return (
		<div
			style={{
				minHeight: "100vh",
				display: "grid",
				placeItems: "center",
				padding: "20px",
			}}>
			<Script
				type='module'
				src='https://unpkg.com/@splinetool/viewer@1.12.74/build/spline-viewer.js'
				strategy='afterInteractive'
			/>

			<section
				className='business-card'
				style={{
					width: "100%",
					maxWidth: "1020px",
					display: "grid",
					gridTemplateColumns: "1.1fr 1fr",
					overflow: "hidden",
				}}>
				<div
					style={{
						padding: "36px",
						borderRight: "1px solid var(--line)",
					}}>
					<p
						className='badge'
						style={{
							display: "inline-block",
							marginBottom: "12px",
						}}>
						Moly AI Chat
					</p>
					<h1
						style={{
							margin: 0,
							fontSize: "30px",
							lineHeight: 1.2,
						}}>
						Panel logowania
					</h1>
					<p
						style={{
							color: "var(--muted)",
							marginTop: "10px",
							marginBottom: "28px",
						}}>
						Profesjonalny dostęp do asystenta biznesowego. Zaloguj
						się, aby przejść do czatu.
					</p>

					<form
						onSubmit={onSubmit}
						style={{ display: "grid", gap: "14px" }}>
						<label
							style={{
								display: "grid",
								gap: "8px",
								fontSize: "14px",
							}}>
							E-mail
							<input
								className='input'
								type='email'
								value={email}
								onChange={(e: ChangeEvent<HTMLInputElement>) =>
									setEmail(e.target.value)
								}
								placeholder='twoj@email.com'
								autoComplete='email'
								required
							/>
						</label>

						<label
							style={{
								display: "grid",
								gap: "8px",
								fontSize: "14px",
							}}>
							Hasło
							<div style={{ position: "relative" }}>
								<input
									className='input'
									style={{ paddingRight: "95px" }}
									type={showPassword ? "text" : "password"}
									value={password}
									onChange={(
										e: ChangeEvent<HTMLInputElement>,
									) => setPassword(e.target.value)}
									placeholder='••••••••••'
									autoComplete='current-password'
									required
								/>
								<button
									type='button'
									className='button-outline'
									onClick={() =>
										setShowPassword(prev => !prev)
									}
									style={{
										position: "absolute",
										right: "6px",
										top: "50%",
										transform: "translateY(-50%)",
										fontSize: "12px",
										padding: "6px 10px",
									}}>
									{showPassword ? "Ukryj" : "Pokaż"}
								</button>
							</div>
						</label>

						{error ? (
							<div
								style={{
									border: "1px solid #fecaca",
									background: "#fef2f2",
									color: "var(--danger)",
									borderRadius: "8px",
									padding: "10px 12px",
									fontSize: "13px",
								}}>
								{error}
							</div>
						) : null}

						<button
							className='button-primary'
							disabled={formDisabled}
							type='submit'>
							<span
								style={{
									display: "inline-flex",
									alignItems: "center",
									gap: "8px",
								}}>
								{loading ? <span className='spinner' /> : null}
								{accentLabel}
							</span>
						</button>
					</form>

					<p
						style={{
							marginTop: "14px",
							color: "#64748b",
							fontSize: "12px",
						}}>
						Polaczenie szyfrowane. Dane przetwarzane po stronie
						backendu.
					</p>
				</div>

				<div
					style={{
						background:
							"linear-gradient(165deg, #eff6ff 0%, #ffffff 65%)",
						minHeight: "500px",
						position: "relative",
					}}>
					<div
						style={{
							position: "absolute",
							top: "14px",
							left: "14px",
							right: "14px",
							border: "1px dashed #bfdbfe",
							borderRadius: "8px",
							padding: "8px 10px",
							color: "#1e40af",
							fontSize: "12px",
							zIndex: 2,
							background: "rgba(255,255,255,0.75)",
						}}>
						Spline scene: {splineSceneUrl}
					</div>
					<div style={{ height: "100%" }}>
						<spline-viewer url={splineSceneUrl}></spline-viewer>
					</div>
				</div>
			</section>

			<style jsx>{`
				@media (max-width: 920px) {
					section {
						grid-template-columns: 1fr !important;
					}

					section > div:first-child {
						border-right: none !important;
						border-bottom: 1px solid var(--line);
					}
				}
			`}</style>
		</div>
	);
}
