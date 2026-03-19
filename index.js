const express = require('express');
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { GoogleAIFileManager } = require("@google/generative-ai/server");
const { YoutubeTranscript } = require("youtube-transcript");
const fs = require("fs");
const multer = require("multer");
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(express.json());
app.use(cors());

// Configure Multer for temporary file storage
const upload = multer({ dest: "uploads/" });

// Initialize Google AI components
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);

const { PDFDocument } = require('pdf-lib');

const GEN_CONFIG = {
    model: "gemini-3.1-flash-lite-preview", // The specific model version
    generationConfig: { 
        responseMimeType: "application/json", // Forces the AI to return valid JSON
        maxOutputTokens: 8192,                // The length limit for the AI's reply
        temperature: 0.7                      // Creativity level (0.7 is good for academic notes)
    }
};

const FILE_PROMPT = `PHASE 1: CONTENT MAPPING (Internal Monologue)
Before generating the JSON, perform a deep-scan of the uploaded file. 
1. Identify every major header and sub-header.
2. For each sub-topic, extract at least 3-5 distinct "raw facts" or "logical steps."
3. Plan how to expand each "note" to be at least 60-100 words of pedagogical detail.
        
      You are an academic content transformation engine specialized in multi-layered educational datasets. Analyze the uploaded link and do the following:

OUTPUT ARCHITECTURE (THE NESTED HEAD)
The output must be a single JSON object with exactly one root key: "fileJson".
The value of "fileJson" must be an array containing multiple objects (minimum 2).
Each object within the fileJson array must contain a "lecture_topic" key, which is an array containing multiple topic objects (minimum 2).

Strict Target Schema:
{
"fileJson": [
{
"lecture_topic": [
{ "topic": "Module 1 - Part A", "notes": [...], "quiz": [...], "notes_N": [...], "flashcards": [...] },
{ "topic": "Module 1 - Part B", "notes": [...], "quiz": [...], "notes_N":[...], "flashcards": [...] }
]
},

]
}

ADAPTIVE SEGMENTATION ENGINE
File-Level Scaling: Group the source material into high-level conceptual modules. Create a new object in the fileJson array for every major shift in subject matter. Ensure at least two such objects exist.

Topic-Level Scaling: Within each file object, break the content down into specific sub-topics. Create as many lecture_topic objects as necessary to cover the material clearly. Ensure at least two per file object.

STRICT CONTENT DENSITY RULES:
1. **Note Volume:** Each "note" object must be comprehensive. Do not summarize. If the source discusses a process, explain every step in the "paragraph" field. 
2. **The 100-Word Target:** Aim for high word counts in the "paragraph" and "important" fields.

Paragraph Labeling: For each note, evaluate paragraphs. Label the most vital core concept or lists of values or contents in the note as "important": "text". If the lists is in the form index. title:- explanation, detect the index. title and wrap it in '<p class="listing-explanation" style="line-height:25px;padding:5px 15px;font-size:16px"><b style="opcity:0.7;fot-size:16.5px;color:#6366f1">index. title:-</b> explanation </p>'. Also label supporting details as "paragraph": "text".

Intros & Conclusions: Every note must include an "intro" establishing relevance and a "conclusion" synthesizing the insights. The conclusion must be engaging, anticipatory, and should lead users to the next note. Rhetorical questions can be used for the conclusion to lead users ti the next note. For example, say the next note would be talking on arms of government and the previous note talked about government, then the conclusion of the previous note can be "Have you heard that there are various divisions of government known as the 'arms of government'?" or 'Did you know that we have the arms of government? Check it out in the next slide'.

Theme: Every note must include a "theme", a general short summary title of the note.

Examples: Add a detailed "example" array to each note.

Requirements for Examples:

Quantity: Provide between 2 and 5 examples per note.

Length: Each example must be a minimum of 30 words.

 examples should come only when needed! Avoid unrequired examples.

Formatting: Use HTML <b> tags to highlight key terms, variables, or core concepts within the text.

Style Guide Example:

Input: "Alternating current is a continuous time signal because x(t) has values for every point in time."

Output: "<b>Alternating current</b> is a continuous time signal because <b>x(t)</b> has values for every point in time."



Quizzes: Insert a "quiz" array at the end of the notes.

Svg_diagrams: To better explain the notes, visuals are required. When necessary, each note should include an "svg_diagram", an html tag inthis form '     
<div style="display:flex;justify-content:center">    
<svg width="95vw" height="160px" viewbox = "0 0 500 150" preserveAspectRatio="xMidYMid meet" style="marin-left:-20px;display:block;"> .. remaining tags .. </svg></div>'. It should be a well labeled 2D shape that better explains the note. fill it with no colors but make the stroke colored: #abbeda, the text color:#abbeda and the stroke-width to be 1. Add a leading diagram explanation paragraph at the bottom. Readers should be able to recall important points whenever they remember the image.

Flashcards: Generate 2–5 concise, concept-focused summary flashcards per lecture_topic.

SOURCE-GROUNDING & VALIDATION
Anti-Hallucination: All generated text must be derived directly from the provided notes. Do not introduce external academic theories, unrelated facts, or invented statistics.

Structural Integrity: Before producing output, verify that JSON syntax is valid and the "Multiple-within-Multiple" structure is strictly maintained.

FINAL OUTPUT RULE
 Return valid JSON only. Do not include markdown code blocks (json ... ). Do not include any introductory text, headers, or footers...  
 If the file is extremely long, prioritize the whole file with maximum detail. 
  Start the response with { and end with }      
      `

/**
 * Helper: Polls the Google File Manager until the file is fully processed.
 */
async function waitForFile(fileName) {
    let file = await fileManager.getFile(fileName);
    while (file.state === "PROCESSING") {
        process.stdout.write("."); // Prints dots in your terminal to show progress
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
        file = await fileManager.getFile(fileName);
    }
    if (file.state === "FAILED") {
        throw new Error("File processing failed on Google servers.");
    }
    return file;
}

app.post('/generate', upload.single('file'), async (req, res) => {
    let tempFiles = [];
    try {
        if (!req.file) return res.status(400).json({ error: "No file uploaded" });

        // 1. Split the PDF into chunks of 500 pages or less
        const existingPdfBytes = fs.readFileSync(req.file.path);
        const pdfDoc = await PDFDocument.load(existingPdfBytes);
        const totalPages = pdfDoc.getPageCount();
        const chunkSize = 500;
        const chunks = [];

        for (let i = 0; i < totalPages; i += chunkSize) {
            const newPdf = await PDFDocument.create();
            const end = Math.min(i + chunkSize, totalPages);
            const pageIndices = Array.from({ length: end - i }, (_, k) => i + k);
            
            const copiedPages = await newPdf.copyPages(pdfDoc, pageIndices);
            copiedPages.forEach(page => newPdf.addPage(page));
            
            const path = `uploads/chunk_${i}.pdf`;
            fs.writeFileSync(path, await newPdf.save());
            chunks.push(path);
            tempFiles.push(path);
        }

        // 2. Process each chunk through Gemini
        const partialResults = [];
        for (const chunkPath of chunks) {
            const uploadResult = await fileManager.uploadFile(chunkPath, {
                mimeType: "application/pdf",
                displayName: "Chunk",
            });

            const file = await waitForFile(uploadResult.file.name);
            const model = genAI.getGenerativeModel(GEN_CONFIG);
            const result = await model.generateContent([
                { fileData: { mimeType: file.mimeType, fileUri: file.uri } },
                { text: FILE_PROMPT } // Your detailed prompt from before
            ]);
            
            partialResults.push(await result.response.text());
        }

        // 3. Final Step: The "Reducer" 
        // We send all JSON strings back to Gemini to merge into one valid structure
        const finalModel = genAI.getGenerativeModel({ model: "gemini-3.1-flash-lite-preview" });
        const mergePrompt = `
            Merge the following JSON arrays into a single valid "fileJson" object. 
            Ensure the "Multiple-within-Multiple" structure is preserved.
            Remove any duplicate topics if they appeared in multiple chunks.
            
            Data to merge: ${JSON.stringify(partialResults)}
        `;

        const finalResult = await finalModel.generateContent(mergePrompt);
        res.json({ analysis: await finalResult.response.text() });

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
    } finally {
        // Cleanup all temp chunks and the original upload
        tempFiles.forEach(f => { if(fs.existsSync(f)) fs.unlinkSync(f); });
        if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    }
});

/**
 * Extracts the 11-character YouTube Video ID from any URL format.
 * Works for: youtube.com, youtu.be, and youtube.com/shorts/
 */
function extractVideoId(url) {
    const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/|youtube\.com\/shorts\/)([^"&?\/\s]{11})/i;
    const match = url.match(regex);
    return match ? match[1] : null;
}

const { youtubeDl } = require('youtube-dl-exec');

app.post('/urlGenerate', async (req, res) => {
    const { videoUrl } = req.body;

    try {
        console.log("Starting extraction for:", videoUrl);

        // 1. Get Video Metadata and Transcript URLs
        // --skip-download: Don't download the actual video file
        // --write-auto-subs: Get auto-generated captions if manual ones are missing
        const info = await youtubeDl(videoUrl, {
            dumpSingleJson: true,
            skipDownload: true,
            writeAutoSubs: true,
            subLang: 'en',
            subFormat: 'json3'
        });

        // 2. Locate the English subtitle URL
        const captions = info.requested_subtitles || info.automatic_captions;
        const englishTrack = captions?.en || captions?.['en-us'] || captions?.['en-gb'];

        if (!englishTrack) {
            return res.status(404).json({ error: "No English transcripts found for this video." });
        }

        // 3. Fetch the actual transcript text from the URL yt-dlp found
        const subResponse = await fetch(englishTrack.url);
        const subData = await subResponse.json();

        // 4. Clean the "JSON3" format into a single string
        // JSON3 puts text in 'events' -> 'segs'
        const fullText = subData.events
            .filter(event => event.segs)
            .map(event => event.segs.map(s => s.utf8).join(''))
            .join(' ')
            .replace(/\s+/g, ' ') // Remove extra spaces
            .substring(0, 20000); // Limit to ~15-20 mins of talk for Gemini

        // 5. Send to Gemini
        const model = genAI.getGenerativeModel({ 
            model: "gemini-3.1-flash-lite-preview",
            generationConfig: { responseMimeType: "application/json",
              maxOutputTokens: 8192
              
            }
        });

        const prompt = `Analyze: ${fullText}
        
        PHASE 1: CONTENT MAPPING (Internal Monologue)
Before generating the JSON, perform a deep-scan of the uploaded file. 
1. Identify every major header and sub-header.
2. For each sub-topic, extract at least 3-5 distinct "raw facts" or "logical steps."
3. Plan how to expand each "note" to be at least 60-100 words of pedagogical detail.
        
      You are an academic content transformation engine specialized in multi-layered educational datasets. Analyze the uploaded link and do the following:

OUTPUT ARCHITECTURE (THE NESTED HEAD)
The output must be a single JSON object with exactly one root key: "fileJson".
The value of "fileJson" must be an array containing multiple objects (minimum 2).
Each object within the fileJson array must contain a "lecture_topic" key, which is an array containing multiple topic objects (minimum 2).

Strict Target Schema:
{
"fileJson": [
{
"lecture_topic": [
{ "topic": "Module 1 - Part A", "notes": [...], "quiz": [...], "notes_N": [...], "flashcards": [...] },
{ "topic": "Module 1 - Part B", "notes": [...], "quiz": [...], "notes_N":[...], "flashcards": [...] }
]
},
{
"lecture_topic": [
{ "topic": "Module 2 - Part A", "notes": [...], "quiz": [...], "notes_N": [...], "flashcards": [...] },
{ "topic": "Module 2 - Part B", "notes": [...], "quiz": [...], "notes_N":[...], "flashcards": [...] }
]
}
]
}

ADAPTIVE SEGMENTATION ENGINE
File-Level Scaling: Group the source material into high-level conceptual modules. Create a new object in the fileJson array for every major shift in subject matter. Ensure at least two such objects exist.

Topic-Level Scaling: Within each file object, break the content down into specific sub-topics. Create as many lecture_topic objects as necessary to cover the material clearly. Ensure at least two per file object.

STRICT CONTENT DENSITY RULES:
1. **Note Volume:** Each "note" object must be comprehensive. Do not summarize. If the source discusses a process, explain every step in the "paragraph" field. 
2. **The 100-Word Target:** Aim for high word counts in the "paragraph" and "important" fields.

Paragraph Labeling: For each note, evaluate paragraphs. Label the most vital core concept or lists of values or contents in the note as "important": "text". If the lists is in the form index. title:- explanation, detect the index. title and wrap it in '<p class="listing-explanation" style="line-height:25px;padding:5px 15px;font-size:16px"><b style="opcity:0.7;fot-size:16.5px;color:#6366f1">index. title:-</b> explanation </p>'. Also label supporting details as "paragraph": "text".

Intros & Conclusions: Every note must include an "intro" establishing relevance and a "conclusion" synthesizing the insights. The conclusion must be engaging, anticipatory, and should lead users to the next note. Rhetorical questions can be used for the conclusion to lead users ti the next note. For example, say the next note would be talking on arms of government and the previous note talked about government, then the conclusion of the previous note can be "Have you heard that there are various divisions of government known as the 'arms of government'?" or 'Did you know that we have the arms of government? Check it out in the next slide'.

Theme: Every note must include a "theme", a general short summary title of the note.

Examples: Add a detailed "example" array to each.

Requirements for Examples:

Quantity: Provide between 2 and 5 examples per note.

Length: Each example must be a minimum of 30 words.

 examples should come only when needed! Avoid unrequired examples.

Formatting: Use HTML <b> tags to highlight key terms, variables, or core concepts within the text.

Style Guide Example:

Input: "Alternating current is a continuous time signal because x(t) has values for every point in time."

Output: "<b>Alternating current</b> is a continuous time signal because <b>x(t)</b> has values for every point in time."



Quizzes: Insert a quiz after every 3–5 notes. Any notes appearing after the quiz must be placed inside the "notes_N": [] array. Note: the examples should come only when needed! Avoid unrequired examples.

Svg_diagrams: To better explain the notes, visuals are required. When necessary, each note should include an "svg_diagram", an html tag in this form '     
<div style="display:flex;justify-content:center">    
<svg width="95vw" height="160px" viewbox = "0 0 500 150" preserveAspectRatio="xMidYMid meet" style="marin-left:-20px;display:block;"> .. remaining tags .. </svg></div>'. It should be a well labeled 2D shape that better explains the note. fill it with no colors but make the stroke colored: #abbeda, the text color:#abbeda and the stroke-width to be 1. Add a leading diagram explanation paragraph at the bottom. Readers should be able to recall important points whenever they remember the image.

Flashcards: Generate 2–5 concise, concept-focused summary flashcards per lecture_topic.

SOURCE-GROUNDING & VALIDATION
Anti-Hallucination: All generated text must be derived directly from the provided notes. Do not introduce external academic theories, unrelated facts, or invented statistics.

Structural Integrity: Before producing output, verify that JSON syntax is valid and the "Multiple-within-Multiple" structure is strictly maintained.

FINAL OUTPUT RULE
 Return valid JSON only. Do not include markdown code blocks (json ... ). Do not include any introductory text, headers, or footers. Start the response with { and end with }..  
 If the file is extremely long, prioritize the whole file with maximum detail. 
        
        `;
        const result = await model.generateContent(prompt);

        const aiResponseText = await result.response.text();
        res.json({ analysis: aiResponseText });
        
        console.log(result)

    } catch (error) {
        console.error("Process Error:", error);
        res.status(500).json({ error: "Failed to process video", details: error.message });
    }
});


const PORT = 10000;
app.listen(PORT,"0.0.0.0", () => {
    console.log(`✅ Server running on http://localhost:${PORT}`);
});
