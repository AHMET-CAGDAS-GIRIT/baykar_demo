import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export const queryRAG = async (payload) => {
  // /api/v1/query yerine main.py'deki doğru endpoint olan /ask yazılıyor:
  const response = await axios.post(`${API_BASE_URL}/ask`, payload);
  return response.data;
};