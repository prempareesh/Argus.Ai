/**
 * Vercel Serverless Function: /api/vision
 * Secure server-side gateway to NVIDIA NIM Multimodal Vision AI
 * (nvidia/nemotron-3.5-super-vl-120b-a12b & fallbacks)
 *
 * Protects private API keys from client-side exposure.
 */

const fs = require('fs');
const path = require('path');

// Auto-load fallback .env locally if not in Vercel production environment
function loadLocalEnv() {
    if (process.env.NVIDIA_API_KEY) return;
    const candidates = [
        path.join(process.cwd(), '.env'),
        path.join(process.cwd(), 'Visual Explanation', 'Visual Explanation', '.env'),
        path.join(__dirname, '..', '.env'),
        path.join(__dirname, '..', 'Visual Explanation', 'Visual Explanation', '.env')
    ];
    for (const p of candidates) {
        if (fs.existsSync(p)) {
            try {
                const lines = fs.readFileSync(p, 'utf-8').split('\n');
                for (const line of lines) {
                    const match = line.match(/^\s*([\w.-]+)\s*=\s*(.*)?\s*$/);
                    if (match) {
                        const key = match[1];
                        let val = (match[2] || '').trim();
                        if (val.startsWith('"') && val.endsWith('"')) val = val.slice(1, -1);
                        if (val.startsWith("'") && val.endsWith("'")) val = val.slice(1, -1);
                        if (!process.env[key]) process.env[key] = val;
                    }
                }
            } catch (e) {
                // Ignore fallback read errors
            }
        }
    }
}

loadLocalEnv();

// Candidate models in priority order
const VISION_MODELS = [
    'nvidia/nemotron-3.5-super-vl-120b-a12b',
    'google/diffusiongemma-26b-a4b-it',
    'meta/llama-3.2-11b-vision-instruct'
];

module.exports = async function handler(req, res) {
    // 1. Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (req.method === 'OPTIONS') {
        return res.status(200).end();
    }

    // Health check endpoint
    if (req.method === 'GET') {
        const hasKey = Boolean(process.env.NVIDIA_API_KEY && process.env.NVIDIA_API_KEY.trim());
        return res.status(200).json({
            status: 'online',
            service: 'Argus Vision AI Gateway',
            primaryModel: VISION_MODELS[0],
            keyConfigured: hasKey,
            supportedMethods: ['POST', 'GET', 'OPTIONS']
        });
    }

    if (req.method !== 'POST') {
        return res.status(405).json({
            success: false,
            error: 'Method Not Allowed. Use POST.'
        });
    }

    // 2. Validate API Key
    const apiKey = process.env.NVIDIA_API_KEY;
    if (!apiKey || apiKey.trim() === '') {
        return res.status(503).json({
            success: false,
            error: 'NVIDIA API key not configured on server. Please set NVIDIA_API_KEY in Vercel Environment Variables.'
        });
    }

    // 3. Parse and Validate Request Payload
    let body = req.body;
    if (typeof body === 'string') {
        try {
            body = JSON.parse(body);
        } catch (e) {
            return res.status(400).json({
                success: false,
                error: 'Invalid JSON body.'
            });
        }
    }

    const { image, mode = 'scene', customPrompt, language = 'en' } = body || {};

    if (!image || typeof image !== 'string') {
        return res.status(400).json({
            success: false,
            error: 'Missing required "image" field. Must be a base64 Data URL (e.g., data:image/jpeg;base64,...).'
        });
    }

    if (!image.startsWith('data:image/')) {
        return res.status(400).json({
            success: false,
            error: 'Invalid image format. Must start with data:image/ (e.g. data:image/jpeg;base64,...).'
        });
    }

    // Payload size guard (~10MB max)
    if (image.length > 14 * 1024 * 1024) {
        return res.status(413).json({
            success: false,
            error: 'Image payload too large. Maximum supported size is 10MB.'
        });
    }

    // 4. Construct System & User Prompt based on Assistive Goal
    let promptText = "";
    if (customPrompt && typeof customPrompt === 'string' && customPrompt.trim().length > 0) {
        promptText = `You are VisionAid AI, an assistive multimodal assistant for a blind individual. Answer the user's question directly and concisely based on the visible evidence in the supplied image. If an object or detail is not visible or cannot be confirmed from the image, explicitly state that it is not visible or unclear. Do not provide lengthy internal thoughts; provide your direct answer immediately.\nUser Question: "${customPrompt.trim()}"`;
    } else if (mode === 'medicine') {
        promptText = "You are VisionAid AI, an assistive medical vision assistant for a blind individual. Directly read and explain this medicine packaging or bottle. State the exact brand or generic medication name, dosage strength, formulation (tablet, capsule, syrup), instructions, and expiration date if visible. If text or numbers are blurred, partially cut off, or illegible, explicitly warn the user that they cannot be confirmed. Provide your direct answer immediately.";
    } else if (mode === 'ocr' || mode === 'document') {
        promptText = "You are VisionAid AI. Directly transcribe all readable text from this document, label, or sign accurately and verbatim. Preserve numbers, dates, and paragraph order. If any section is blurry, obstructed, or unreadable, explicitly state that the text cannot be read clearly. Provide your direct answer immediately.";
    } else {
        // General scene & object understanding
        promptText = "You are VisionAid AI, an assistive vision guide helping a blind person understand their immediate environment. Describe this scene clearly and concisely in 2 to 3 spoken sentences. Identify prominent objects, their relative positions (left, center, right, foreground), and distinguish visually similar objects (for instance, distinguish an AC remote from a smartphone by checking for physical rubber buttons vs a glass touchscreen). Highlight any obstacles ahead. Provide your direct spoken description immediately without internal thought preamble.";
    }

    // Append localized response directive when Telugu or Hindi is selected
    if (language === 'te' || language === 'te-IN') {
        promptText += "\n\nImportant Directive: Directly respond completely and naturally in Telugu (తెలుగు). Use 2 to 3 clear, helpful spoken sentences.";
    } else if (language === 'hi' || language === 'hi-IN') {
        promptText += "\n\nImportant Directive: Directly respond completely and naturally in Hindi (हिन्दी). Use 2 to 3 clear, helpful spoken sentences.";
    }

    // 5. Query NVIDIA NIM Multimodal Vision API with Fallback Hierarchy
    const endpoint = 'https://integrate.api.nvidia.com/v1/chat/completions';
    let lastError = null;

    for (const modelName of VISION_MODELS) {
        const payload = {
            model: modelName,
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: promptText },
                        { type: 'image_url', image_url: { url: image } }
                    ]
                }
            ],
            max_tokens: 1024,
            temperature: 0.15
        };

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 28000); // 28 second timeout per attempt

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey.trim()}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload),
                signal: controller.signal
            });

            clearTimeout(timeout);

            if (!response.ok) {
                const errText = await response.text();
                console.warn(`[NVIDIA NIM Error] Model ${modelName} returned status ${response.status}:`, errText.slice(0, 150));
                lastError = {
                    status: response.status,
                    message: `Vision model ${modelName} returned status ${response.status}.`,
                    details: errText.slice(0, 200)
                };
                // Try next candidate model
                continue;
            }

            const data = await response.json();
            const message = data?.choices?.[0]?.message;
            let content = message?.content;

            // Handle reasoning models where output might reside in reasoning_content
            if (!content || typeof content !== 'string' || content.trim() === '') {
                if (message?.reasoning_content && typeof message.reasoning_content === 'string') {
                    const reasoning = message.reasoning_content.trim();
                    const paragraphs = reasoning.split(/\n\s*\n/).filter(p => p.trim().length > 0);
                    if (paragraphs.length > 0) {
                        content = paragraphs[paragraphs.length - 1].replace(/^[\s*#-]+/gm, '').trim();
                    }
                }
            }

            if (!content || content.trim() === '') {
                console.warn(`[NVIDIA NIM] Model ${modelName} returned empty text, trying next model.`);
                lastError = {
                    status: 502,
                    message: `Vision model ${modelName} returned an empty response.`
                };
                continue;
            }

            const cleanDescription = content.trim();

            return res.status(200).json({
                success: true,
                description: cleanDescription,
                model: modelName,
                mode
            });

        } catch (err) {
            clearTimeout(timeout);
            console.warn(`[Vision API Exception] Model ${modelName}:`, err.message);
            lastError = {
                status: err.name === 'AbortError' ? 504 : 502,
                message: err.name === 'AbortError' ? 'Vision AI request timed out.' : err.message
            };
        }
    }

    // If all upstream vision models failed, return HTTP 502/504 (NEVER return 404!)
    const returnStatus = lastError?.status === 504 ? 504 : 502;
    return res.status(returnStatus).json({
        success: false,
        error: lastError?.message || 'Vision AI service currently unavailable. Please try capturing the frame again.',
        details: lastError?.details || null
    });
};
