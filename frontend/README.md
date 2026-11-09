# DeepTrace frontend

React + Vite frontend for the existing FastAPI analysis service.

## Run locally

From the project root, start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. Vite proxies analysis requests to port 8000. Samples, fonts, icons, and artwork are bundled locally.

## Production build

```powershell
cd frontend
npm run build
```

Start or restart the FastAPI server after the first build. It serves `dist/index.html` and the generated `/assets` files at http://localhost:8000. `npm run preview` also proxies the local backend.

For a separately hosted API, set `VITE_API_BASE_URL` before building. With the same-origin FastAPI setup, no frontend environment variables are needed.

## On a phone

Run Vite with `npm run dev -- --host 0.0.0.0` and open the computer’s LAN address on port 3000 from a phone on the same network. The API can stay bound to localhost because Vite proxies its requests. The host firewall must permit the development server.

## Behavior

- Hash navigation supports reloads and browser back/forward without server rewrite rules.
- Theme follows the operating system on first visit, then remembers the selected light/dark mode.
- Uploads accept JPEG, PNG, WebP, and BMP up to 10 MB. Corrupt images and images over 40 megapixels are rejected.
- Results display the backend’s primary fused verdict separately from supplementary specialist votes. No predictions or progress percentages are fabricated.
- Multiple faces can be selected in results, evidence, and diagnostics. Reports include every face.
- The browser stores full reports in IndexedDB, including images. Older localStorage reports are migrated. Storage failures leave analysis and downloads available and display a message.
- PDF generation is lazy-loaded. JSON exports preserve the complete response. Print has its own light layout.
- Cancel aborts the client request; it does not promise to stop inference already running on the server. Requests time out after three minutes.

## Artwork and fonts

See [asset credits](src/assets/CREDITS.md). No third-party image or font service is required at runtime.
