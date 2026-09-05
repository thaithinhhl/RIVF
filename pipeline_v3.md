# WISE-v3 — Witness Identification and Selection of Evidence

WISE-v3 là pipeline thống nhất thay cho routing và minimal-witness heuristic
của v2. Một câu hỏi chỉ đi qua đúng một policy:

```text
question
  → claim/dependency DAG
  → claim-aware semantic seeds
  → bounded directed typed graph proposal
  → topologically conditioned claim-support verification
  → dependency-path verification
  → verified atomic-quote extraction
  → exact minimum serialized-cost witness
  → reasoning-order serialization
```

Không có direct/chained branch, semantic fallback, graph rescue, witness fill,
hay policy thứ hai. Nếu không chứng minh được một witness đầy đủ, pipeline trả
về lỗi và dừng run.

## 1. Reasoning contract

Planner sinh đúng một JSON object chỉ gồm:

- `claims`: các fact nguyên tử cần evidence;
- `operation`: phép `lookup`, `compare`, `boolean`, hoặc `intersection` thực
  hiện sau khi các claim đã được chứng minh.

Planner không sinh `dependencies` hay `source_title`. WISE ground deterministic
mỗi named subject vào exact/unique alias passage title, rồi tạo dependency duy
nhất khi subject variable của claim sau bằng chính xác object variable của một
claim trước. Vì vậy malformed/missing/phantom LLM edges không còn nằm trong
không gian output; đây là canonical construction của một policy, không phải
repair hoặc retry.

`operation.inputs` phải bằng chính xác toàn bộ sink claims của DAG;
output của non-lookup được canonicalize deterministic thành `?answer` để tên
biến tùy ý không bias answer stage. `lookup` có đúng một input, còn
`compare` và `intersection` phải có ít nhất hai. Vì vậy operation không còn là
metadata: plan thiếu một nhánh so sánh hoặc chứa orphan terminal claim sẽ bị
`plan_invalid` trước retrieval. Với `lookup`, output còn phải đúng bằng object
variable của terminal claim.

Planner giữ thứ tự trái-sang-phải của các alternative trong comparison và
phân rã các quan hệ composition (`grandparent`, `in-law`, `stepchild`) thành
các atomic kinship claims dùng chung variable, thay vì tạo direct relation mà
passages không biểu diễn trực tiếp.

Schema thừa/thiếu field, variable-subject không reuse đúng một earlier object,
orphan sink, guessed terminal constant, hoặc operation không khớp đều là
`plan_invalid`; hệ thống không sinh plan thứ hai.

## 2. Typed evidence graph

Node là sentence. Directed edges giữ đồng thời mọi type hợp lệ giữa hai node:

- `same_passage_next` và `same_passage_previous`;
- `entity_coreference`;
- `mentions_title`.

`mentions_title` và binding transport dùng canonical entity aliases, gồm
disambiguation suffix, accent/punctuation/spacing, honorific, và distinctive
name-token overlap (`Cuthwine` ↔ `Cuthwine of Wessex`). Alias chỉ mở graph và
khớp binding; semantic verifier vẫn phải chứng minh relation.

Question và từng claim query tạo semantic seeds. Các claim đã grounded vào
title đưa toàn bộ sentences của title đó vào seed set. Candidate pool là hợp
của các node trong `max_hops` trên graph proposal; đây là một proposal duy
nhất, không có dense rescue nếu pool thiếu evidence.

## 3. Entailment and binding verification

Cross-encoder chỉ prefilter tối đa `claim_prefilter_k` candidates cho mỗi
claim; relevance score không được dùng làm bằng chứng. Claims được verify theo
topological order. Với claim có incoming dependency, candidate title bắt buộc
phải bằng binding value đã verify ở parent; vì vậy hop sau không thể tự gắn vào
một entity distractor dù sentence đó có cùng relation. Đây là một inference
policy duy nhất, không phải rescue pass.

Structured verifier trả sparse positives: mọi `entailed` task phải xuất hiện;
negative task có thể ghi `not_entailed` hoặc omit. `entailed` chỉ hợp lệ
khi sentence cùng passage title thực sự assert subject–relation–object, verifier
resolve chính xác mọi biến, và trả một `evidence_quote` là substring liên tục,
nguyên văn, ngắn nhất đủ chứng minh claim. Quote proposal được chiếu
deterministic theo token order lên source và mở rộng để chứa mọi unconstrained
non-terminal binding; kết quả cuối luôn là một contiguous exact substring.
Entity binding được transport sang hop sau vẫn phải lexical/alias-grounded.
Riêng terminal typed attribute được phép dùng canonical value khi quote biểu
diễn nó bằng morphology chuẩn (`Canadian` → `Canada`); quote extractive vẫn là
certificate và verifier không được dùng outside knowledge. Token paraphrase
không align, binding như `unknown`, hoặc thiếu value đều bị reject.
Chỉ tối đa `claim_top_k` assignments vượt calibrated
`claim_support_threshold` được giữ.

Một dependency chỉ được đề xuất khi source assignment và target assignment
resolve cùng biến thành cùng một normalized value. WISE enumerate tối đa
`dependency_path_top_k` directed shortest-simple typed paths cho mỗi compatible
pair, giới hạn bởi `max_path_hops`; không còn khóa cứng vào một shortest path.
Structured path verifier trả `valid` chỉ khi path jointly entails hai claims,
đúng thứ tự, và entity value thực sự instantiate source object lẫn target
subject. Mere co-occurrence bị xem là invalid.

Trước semantic verification, path còn phải ground biến binding thành ít nhất
một entity surface cụ thể: target passage title trên cạnh `mentions_title`,
hoặc entity giao nhau trên cạnh `entity_coreference` (path một node dùng named
entities ngay trong sentence đó). Value đã resolve phải xuất hiện trong tập
grounding này và được log dưới `binding_value`. Connectivity thuần túy qua cạnh
thứ tự câu không được xem là đã chứng minh entity binding.

Confidence từ structured verifier được temperature-scale trước mọi threshold:

```text
p_cal = sigmoid(logit(p_verifier) / confidence_temperature)
```

`confidence_temperature` và ba thresholds chỉ được chọn trên development
manifest rồi freeze. Giá trị `1.0` trong config hiện là identity starting point
cho development, không được mô tả là empirically calibrated cho tới khi fit.

Một claim không có assignment là `claim_uncovered`. Một dependency không có
assignment pair/path vượt ngưỡng là `dependency_unverified`.

## 4. Exact witness objective

Trên các assignments và paths đã verify, solver duyệt chính xác không gian
certificate còn lại:

```text
minimize  tokens(serialize(W))
subject to
  every claim has exactly one retained assignment
  every DAG dependency has one verified directed path
  every operation input is a verified DAG sink claim
  min confidence(assignments, paths) >= tau_suff
  union(verified assignment quotes, required raw bridge nodes) = W
  tokens(serialize(W)) <= Bmax
```

`tau_suff = witness_sufficiency_threshold` là hard quality constraint. Vì vậy
một witness ngắn nhưng confidence thấp không thể thắng witness dài hơn đạt
sufficiency. Chỉ sau khi thỏa quality constraint, objective mới minimize cost.

Endpoint evidence được serialize bằng verified atomic quote thay vì toàn bộ
source sentence; một path-only bridge node chưa có claim assignment vẫn giữ
nguyên câu để không tạo evidence suy diễn. Token cost dùng đúng compact
reasoning-order context này, gồm title headers. Supporting-fact identity vẫn là
key `(title, sent_id)` của source sentence, nên SP evaluation không đổi. Khi
nhiều witness cùng cost, solver chọn bottleneck verification confidence cao
hơn, sau đó dùng sentence keys để tie-break deterministic. `max_search_states`
chỉ là guard: chạm guard trả `search_space_exhausted`, không chuyển sang greedy.

Certificate cuối phải claim-complete, dependency-complete và không chứa node
ngoài union của assignments/paths. Vì objective được giải chính xác trên
verified search space, witness được log là `wise_certified` và
`wise_irreducible`. Không có feasible certificate dưới budget là
`no_feasible_witness`, không trả partial context.

## 5. Operation-aware answering

Answer stage nhận verified plan, chosen claim assignments và dependency
certificate cùng atomic-quote context. Prompt buộc trace từng operation input,
giữ đúng hướng younger/older/earlier/later, và trả winning option thay vì
yes/no cho câu dạng `A or B`. Answer renderer giữ demonym cho câu hỏi
nationality và chỉ dùng country noun khi câu hỏi yêu cầu rõ named country;
không còn post-hoc demonym→country rewrite làm thay đổi prediction. Không có
generic-answer fallback nếu certificate lỗi.

## 6. Errors and logged diagnostics

CLI in một JSON error và exit code 2 cho các lỗi WISE-v3. Các run thành công
log plan, resolved bindings, raw và calibrated verifier confidence, path rank,
verified dependency paths, sufficiency threshold, serialized token cost,
bottleneck confidence, số search states, provenance của planner và hai
verifier stages, cùng certificate flags trong `results.jsonl`.

Các error code chính:

| Code | Ý nghĩa |
|---|---|
| `configuration_invalid` | tham số sai hoặc có fallback/rescue policy |
| `plan_invalid` | planner output vi phạm reasoning contract |
| `candidate_empty` | proposal không có node |
| `claim_uncovered` | thiếu verified evidence cho claim |
| `claim_verification_invalid` | structured entailment output sai contract |
| `dependency_unverified` | thiếu verified path/binding transition |
| `dependency_verification_invalid` | structured path output sai contract |
| `no_feasible_witness` | không có certificate đầy đủ dưới budget |
| `search_space_exhausted` | exact search vượt guard |
| `certificate_invalid` | internal certificate invariant bị vi phạm |

## 7. Development commands

```bash
python -m rivf.experiments.run_experiment configs/wise_v3_hotpot_development_openrouter.yaml --preflight
python -m rivf.experiments.run_experiment configs/wise_v3_hotpot_development_openrouter.yaml

python -m rivf.experiments.run_experiment configs/wise_v3_2wiki_development_openrouter.yaml --preflight
python -m rivf.experiments.run_experiment configs/wise_v3_2wiki_development_openrouter.yaml
```

`call_llm: false` chỉ tắt answer generation. Retrieval method vẫn gọi planner
một lần, claim verifier đúng một lần cho mỗi planned claim theo topological
order, và dependency verifier tối đa một lần cho mỗi question. Question không
có dependency sẽ ghi verifier stage đó là `skipped`.
