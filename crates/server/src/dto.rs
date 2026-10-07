use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
	Text(String),
	Texts(Vec<String>),
	TokenRow(Vec<u32>),
	TokenRows(Vec<Vec<u32>>),
}

impl EmbeddingInput {
	pub fn into_parts(self) -> (Vec<String>, Vec<Vec<u32>>) {
		match self {
			EmbeddingInput::Text(t) => (vec![t], Vec::new()),
			EmbeddingInput::Texts(v) => (v, Vec::new()),
			EmbeddingInput::TokenRow(r) => (Vec::new(), vec![r]),
			EmbeddingInput::TokenRows(v) => (Vec::new(), v),
		}
	}
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
	pub model: Option<String>,
	pub input: EmbeddingInput,
	pub dimensions: Option<usize>,
	/// "float" (default) or "base64"
	pub encoding_format: Option<String>,
	#[allow(dead_code)]
	pub user: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingItem {
	pub object: &'static str,
	pub index: usize,
	/// Owned float vec / pre-encoded base64: the derive serializes f32 slices
	/// in a single streaming pass (no intermediate serde_json::Value tree).
	pub embedding: EmbeddingData,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingData {
	Floats(Vec<f32>),
	Base64(String),
}

#[derive(Debug, Serialize)]
pub struct Usage {
	pub prompt_tokens: usize,
	pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
	pub object: &'static str,
	pub data: Vec<EmbeddingItem>,
	pub model: String,
	pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct RerankRequest {
	pub model: Option<String>,
	pub query: String,
	pub documents: Vec<String>,
	pub top_n: Option<usize>,
	/// Include document text in the response (vLLM default: true).
	pub return_documents: Option<bool>,
	/// Split long documents into up to this many chunks; score = best chunk.
	pub max_chunks_per_doc: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct RerankDocument {
	pub text: String,
}

#[derive(Debug, Serialize)]
pub struct RerankResult {
	pub index: usize,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub document: Option<RerankDocument>,
	pub relevance_score: f64,
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
	pub id: String,
	pub model: String,
	pub results: Vec<RerankResult>,
	pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TextList {
	One(String),
	Many(Vec<String>),
}

impl TextList {
	pub fn into_vec(self) -> Vec<String> {
		match self {
			TextList::One(s) => vec![s],
			TextList::Many(v) => v,
		}
	}
}

/// TEI-compatible embeddings request.
#[derive(Debug, Deserialize)]
pub struct TeiEmbedRequest {
	pub inputs: Vec<String>,
	#[allow(dead_code)]
	pub normalize: Option<bool>,
}

/// vLLM /v1/score: all pairs between text_1 and text_2.
#[derive(Debug, Deserialize)]
pub struct ScoreRequest {
	pub model: Option<String>,
	pub text_1: TextList,
	pub text_2: TextList,
}

#[derive(Debug, Serialize)]
pub struct ScoreItem {
	pub object: &'static str,
	pub index: usize,
	pub text_1: String,
	pub text_2: String,
	pub score: f64,
}

#[derive(Debug, Serialize)]
pub struct ScoreResponse {
	pub object: &'static str,
	pub data: Vec<ScoreItem>,
	pub model: String,
	pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
	pub model: Option<String>,
	pub input: TextList,
	pub candidate_labels: Vec<String>,
	pub multi_label: Option<bool>,
	#[allow(dead_code)]
	pub hypothesis_template: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ClassifyResult {
	pub labels: Vec<String>,
	pub scores: Vec<f64>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ClassifyResponseBody {
	Single {
		object: &'static str,
		model: String,
		#[serde(flatten)]
		result: ClassifyResult,
		usage: Usage,
	},
	Multiple {
		object: &'static str,
		model: String,
		results: Vec<ClassifyResult>,
		usage: Usage,
	},
}

#[derive(Debug, Deserialize)]
pub struct TrueFalseRequest {
	pub model: Option<String>,
	/// Text (or list of texts) to classify.
	pub input: TextList,
	/// The question, e.g. "Is this email important?" — rephrased into an
	/// assertion for NLI. Optional when `assertion` is given.
	pub question: Option<String>,
	/// Explicit affirmative assertion to test against each input, e.g.
	/// "This email is important." — takes precedence over `question`.
	pub assertion: Option<String>,
	/// Decision threshold for `is_true` (default 0.5).
	pub threshold: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct TrueFalseResult {
	pub label: &'static str,
	pub is_true: bool,
	/// P(input entails the assertion).
	pub true_probability: f64,
	/// The assertion that was judged.
	pub assertion: String,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum TrueFalseResponseBody {
	Single {
		object: &'static str,
		model: String,
		#[serde(flatten)]
		result: TrueFalseResult,
		usage: Usage,
	},
	Multiple {
		object: &'static str,
		model: String,
		results: Vec<TrueFalseResult>,
		usage: Usage,
	},
}

#[derive(Debug, Deserialize)]
pub struct PiiRequest {
	pub model: Option<String>,
	#[serde(default)]
	pub text: Option<String>,
	#[serde(default)]
	pub texts: Option<Vec<String>>,
	pub threshold: Option<f64>,
	/// Keep only these entity types.
	pub types: Option<Vec<String>>,
}


#[derive(Debug, Serialize)]
pub struct EntityOut {
	pub entity_type: String,
	pub text: String,
	pub score: f64,
	pub start: usize,
	pub end: usize,
}

impl From<rsinfer_core::pipeline::pii::Entity> for EntityOut {
	fn from(e: rsinfer_core::pipeline::pii::Entity) -> Self {
		Self {
			entity_type: e.entity_type,
			text: e.text,
			score: e.score,
			start: e.start,
			end: e.end,
		}
	}
}

#[derive(Debug, Serialize)]
pub struct PiiDetectResult {
	pub entities: Vec<EntityOut>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub text: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PiiDetectResponse {
	pub object: &'static str,
	pub model: String,
	pub results: Vec<PiiDetectResult>,
	pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct PiiRedactRequest {
	#[serde(flatten)]
	pub detect: PiiRequest,
	/// "mask" (default) or "remove"
	pub mode: Option<String>,
	pub mask_char: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PiiRedactResult {
	pub text: String,
	pub entities: Vec<EntityOut>,
}

#[derive(Debug, Serialize)]
pub struct PiiRedactResponse {
	pub object: &'static str,
	pub model: String,
	pub results: Vec<PiiRedactResult>,
	pub usage: Usage,
}
