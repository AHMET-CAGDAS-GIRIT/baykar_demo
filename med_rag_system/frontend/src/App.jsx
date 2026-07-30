import React, { useState } from 'react';
import axios from 'axios';

function App() {
	const [question, setQuestion] = useState('');
	const [topK, setTopK] = useState(3);
	const [selectedFileBase64, setSelectedFileBase64] = useState(null);
	const [selectedFileType, setSelectedFileType] = useState(null); // 'image' veya 'pdf'
	const [fileName, setFileName] = useState('');
	const [loading, setLoading] = useState(false);
	const [result, setResult] = useState(null);
	const [error, setError] = useState(null);

	// Gerçek backend bağlantı adresi
	const API_URL = "http://127.0.0.1:8000";

	// Dosyayı Base64 formatına çeviren yardımcı fonksiyon
	const handleFileChange = (e) => {
		const file = e.target.files[0];
		if (!file) return;

		setFileName(file.name);
		
		const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
		const isImage = file.type.startsWith('image/') || /\.(jpg|jpeg|png)$/i.test(file.name);

		if (isPdf) {
			setSelectedFileType('pdf');
		} else if (isImage) {
			setSelectedFileType('image');
		} else {
			setSelectedFileType(null);
			setError("Yalnızca PDF veya görsel (JPG, PNG) dosyaları yüklenebilir.");
			return;
		}

		const reader = new FileReader();
		reader.onloadend = () => {
			setSelectedFileBase64(reader.result);
		};
		reader.readAsDataURL(file);
	};

	const handleSubmit = async (e) => {
		e.preventDefault();
		if (!question.trim()) return;

		setLoading(true);
		setError(null);
		setResult(null);

		try {
			const payload = {
				question: question,
				top_k: Number(topK),
				image_base64: selectedFileType === 'image' ? selectedFileBase64 : null,
				pdf_base64: selectedFileType === 'pdf' ? selectedFileBase64 : null,
			};

			const response = await axios.post(`${API_URL}/query`, payload, {
				headers: {
					'Content-Type': 'application/json',
				},
			});

			setResult(response.data);
		} catch (err) {
			console.error("API Bağlantı Hatası:", err);
			const errorDetail = err.response?.data?.detail || "Sunucuya bağlanırken bir hata oluştu veya RAG motoru yanıt vermedi.";
			setError(errorDetail);
		} finally {
			setLoading(false);
		}
	};

	const getConfidenceBadge = (level) => {
		switch (level) {
			case 'HIGH':
				return 'bg-green-900 text-green-300 border-green-700';
			case 'MEDIUM':
				return 'bg-yellow-900 text-yellow-300 border-yellow-700';
			case 'LOW':
			default:
				return 'bg-red-900 text-red-300 border-red-700';
		}
	};

	return (
		<div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col justify-between">
			<header className="bg-gray-800 border-b border-gray-700 py-4 px-6 shadow-md">
				<div className="max-w-5xl mx-auto flex justify-between items-center">
					<h1 className="text-xl font-bold text-teal-400">Medikal RAG Asistanı</h1>
					<span className="text-xs bg-teal-900 text-teal-300 px-3 py-1 rounded-full border border-teal-700">
						Qwen2 & FAISS Destekli Multimodal Sistem
					</span>
				</div>
			</header>

			<main className="max-w-5xl w-full mx-auto p-6 flex-grow flex flex-col gap-6">
				<form onSubmit={handleSubmit} className="bg-gray-800 p-4 rounded-xl border border-gray-700 shadow-lg flex flex-col gap-4">
					<div className="flex flex-col gap-1.5">
						<label className="text-sm font-medium text-gray-300">Medikal Sorunuzu veya Araştırma Konusunu Girin:</label>
						<input
							type="text"
							value={question}
							onChange={(e) => setQuestion(e.target.value)}
							placeholder="Örn: Göğüs röntgeninde pnömoni bulgusu var mı?"
							className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2.5 text-gray-100 focus:outline-none focus:border-teal-500"
						/>
					</div>

					<div className="flex flex-col md:flex-row gap-4 items-center justify-between">
						<div className="flex flex-col gap-1 w-full md:w-auto flex-grow">
							<label className="text-xs text-gray-400">Opsiyonel Dosya Yükle (Görsel veya PDF):</label>
							<input
								type="file"
								accept="image/*,application/pdf"
								onChange={handleFileChange}
								className="text-xs text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-teal-900 file:text-teal-300 hover:file:bg-teal-800 cursor-pointer bg-gray-900 border border-gray-700 rounded-lg px-2 py-1"
							/>
							{fileName && (
								<span className="text-xs text-teal-400 mt-1">Seçilen Dosya: {fileName}</span>
							)}
						</div>

						<div className="flex items-center gap-3 w-full md:w-auto justify-end">
							<div className="flex flex-col gap-1">
								<span className="text-xs text-gray-400 whitespace-nowrap">Kaynak sayısı:</span>
								<input
									type="number"
									min="1"
									max="10"
									value={topK}
									onChange={(e) => setTopK(e.target.value)}
									className="w-16 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-gray-100 text-center focus:outline-none focus:border-teal-500"
								/>
							</div>
							<button
								type="submit"
								disabled={loading}
								className="bg-teal-600 hover:bg-teal-500 text-white font-medium px-6 py-2.5 rounded-lg transition-colors disabled:opacity-50 cursor-pointer whitespace-nowrap mt-5"
							>
								{loading ? 'Analiz Ediliyor...' : 'Sorgula'}
							</button>
						</div>
					</div>
				</form>

				{error && (
					<div className="bg-red-900/50 border border-red-700 text-red-200 p-4 rounded-xl">
						{error}
					</div>
				)}

				{result && (
					<div className="flex flex-col gap-6">
						<div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-lg">
							<div className="flex flex-wrap justify-between items-center gap-3 mb-3">
								<h2 className="text-lg font-semibold text-teal-400">Yapay Zeka Yanıtı</h2>
								<div className="flex flex-wrap gap-2 items-center">
									<span className={`text-xs px-2.5 py-1 rounded border font-mono ${getConfidenceBadge(result.confidence_level)}`}>
										Güven: {result.confidence_level} ({(result.confidence_score * 100).toFixed(0)}%)
									</span>
									<span className="text-xs text-gray-400 bg-gray-900 px-2.5 py-1 rounded border border-gray-700 font-mono">
										Gecikme: {result.latency_ms} ms
									</span>
								</div>
							</div>
							<p className="text-gray-200 whitespace-pre-wrap leading-relaxed">{result.answer}</p>
							
							{result.warnings && result.warnings.length > 0 && (
								<div className="mt-4 p-3 bg-yellow-900/30 border border-yellow-700/50 rounded-lg text-yellow-300 text-sm">
									<strong>Uyarılar:</strong> {result.warnings.join(", ")}
								</div>
							)}
						</div>

						<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
							<div className="bg-gray-800 p-5 rounded-xl border border-gray-700 shadow-lg">
								<h3 className="text-md font-semibold text-teal-400 mb-3">Erişilen Kanıtlar (Evidence)</h3>
								{result.evidence && result.evidence.length > 0 ? (
									<ul className="space-y-3">
										{result.evidence.map((item, index) => (
											<li key={index} className="text-xs bg-gray-900 p-3 rounded border border-gray-700 flex flex-col gap-2">
												<div className="flex justify-between text-gray-400 font-mono">
													<span className="truncate max-w-[150px]">Dosya: {item.source_file}</span>
													<span>Skor: {item.score ? item.score.toFixed(4) : 'N/A'}</span>
													<span className="text-teal-300">[{item.modalite}]</span>
												</div>

												{item.thumbnail_url && (
													<div className="my-1 flex justify-center bg-gray-950 p-2 rounded border border-gray-800">
														<img 
															src={`${API_URL}${item.thumbnail_url}`} 
															alt={item.source_file}
															className="max-h-32 object-contain rounded"
															onError={(e) => { e.target.style.display = 'none'; }}
														/>
													</div>
												)}

												<p className="text-gray-300 leading-relaxed">{item.snippet}</p>
											</li>
										))}
									</ul>
								) : (
									<p className="text-sm text-gray-400">Bu sorgu için kanıt detayı dönülmedi.</p>
								)}
							</div>

							<div className="bg-gray-800 p-5 rounded-xl border border-gray-700 shadow-lg">
								<h3 className="text-md font-semibold text-teal-400 mb-3">İlgili Görseller ve Kaynak Konumu</h3>
								{result.evidence && result.evidence.some(item => item.thumbnail_url || item.modalite !== 'text') ? (
									<div className="grid grid-cols-2 gap-3">
										{result.evidence
											.filter(item => item.thumbnail_url || item.modalite !== 'text')
											.map((item, index) => (
												<div key={index} className="bg-gray-900 border border-gray-700 rounded p-2 flex flex-col items-center justify-center">
													<span className="text-xs text-gray-400 truncate w-full text-center mb-1">{item.source_file}</span>
													{item.thumbnail_url ? (
														<img 
															src={`${API_URL}${item.thumbnail_url}`} 
															alt={item.source_file}
															className="max-h-24 object-contain rounded border border-gray-800"
															onError={(e) => { e.target.style.display = 'none'; }}
														/>
													) : (
														<span className="text-xs text-gray-500 italic py-6">Görsel Alanı</span>
													)}
												</div>
											))}
									</div>
								) : (
									<p className="text-sm text-gray-400">Bu sorgu için referans görsel bulunamadı.</p>
								)}
							</div>
						</div>
					</div>
				)}
			</main>

			<footer className="bg-gray-800 border-t border-gray-700 py-3 px-6 text-center text-xs text-gray-500">
				Medikal Sorumluluk Reddi: Bu sistem yalnızca araştırma ve demo amaçlıdır. Kesin tanı veya tedavi amacıyla kullanılmamalıdır.
			</footer>
		</div>
	);
}

export default App;