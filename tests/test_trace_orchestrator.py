import asyncio
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.trace_orchestrator import (
    branch_candidates, context, ensure_layer_inventory, fork,
    seed_branch_cases, deduplicate_layer_inventory, deduplicate_trace_cases,
    enqueue_layers, _persist_trace_case, scan_layers, _trace_case,
    prune_ineligible_optimization_cases,
    next_class_method, next_unvisited_branch, prune_ineligible_branch_tasks,
    prune_invalid_roots, reset, run, sass_optimization_dependent,
    sass_sequential_successor, setup, validate_step, verified_linear_binding_target, linear_binding_candidate, linear_addmm_candidate,
    retry_unresolved_validation_errors,
    imported_base_class_target, same_file_call_target, native_branch_nodes,
    retry_unresolved_navigation_candidates,
)


class TraceOrchestratorTests(unittest.TestCase):

    def test_exported_layers_with_imported_bases_are_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = Path(tmp) / "sources"
            keras = sources / "keras" / "keras" / "src" / "layers" / "rnn" / "lstm.py"
            keras.parent.mkdir(parents=True)
            keras.write_text("\n".join([
                "@keras_export('keras.layers.LSTM')",
                "class LSTM(RNN):",
                "    pass",
                "@keras_export(['keras.layers.Conv2D', 'keras.layers.Convolution2D'])",
                "class Conv2D(BaseConv):",
                "    pass",
            ]), encoding="utf-8")
            tensorflow = sources / "tensorflow" / "python" / "layers" / "exported.py"
            tensorflow.parent.mkdir(parents=True)
            tensorflow.write_text("\n".join([
                "@tf_export('keras.layers.Exported')",
                "class Exported(ImportedLayer):",
                "    pass",
            ]), encoding="utf-8")

            rows = list(scan_layers(sources))
            found = {(row[1], row[2], row[3]) for row in rows}
            self.assertIn(("keras", "LSTM", "keras/src/layers/rnn/lstm.py"), found)
            self.assertIn(("keras", "Conv2D", "keras/src/layers/rnn/lstm.py"), found)
            self.assertIn(("tensorflow", "Exported", "python/layers/exported.py"), found)
            self.assertEqual(len([row for row in rows if row[2] == "LSTM"]), 1)

    def fixture(self, base):
        sources = base / "sources"
        dense = sources / "keras" / "keras" / "layers" / "dense.py"
        dense.parent.mkdir(parents=True)
        dense.write_text(
            "class Dense(Layer):\n"
            "    def call(self, x):\n"
            "        if self.use_bias:\n"
            "            return x + 1\n"
            "        else:\n"
            "            return x\n"
        )
        db = base / "cases.sqlite"
        with sqlite3.connect(db) as conn:
            setup(conn)
            conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                         ("dense", "keras", "Dense", "keras/layers/dense.py", 1,
                          hashlib.sha256(dense.read_bytes()).hexdigest(), "layer_class", "pending"))
            conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                         ("dense-root", "dense", "keras/layers/dense.py", 1, "{}"))
        return sources, db, dense

    def model(self, stage, payload, prompt):
        if prompt == "planner":
            return {"focus": "follow source control flow"}
        if prompt == "agent_trace_worker":
            line = payload["line"]
            source_line = next(x.split(": ", 1)[1] for x in payload["source"].splitlines()
                               if x.startswith(f"{line}: "))
            return {"claims": [{"line": line, "quote": source_line, "claim": "quoted source"}],
                    "next": None, "terminal": "unresolved"}
        return {"approved_indices": [0]}

    def test_claim_quote_must_be_visible_when_source_line_is_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=root/"sources"
            src=sources/"keras"/"keras"/"layers"/"long.py"
            src.parent.mkdir(parents=True)
            src.write_text("A"*350+"\n")
            ctx=context(sources,"keras","keras/layers/long.py",1)
            self.assertIn("[line truncated]",ctx["source"])
            self.assertEqual(len(ctx["visible_lines"][1]),300)
            with self.assertRaisesRegex(ValueError,"claim lacks exact source evidence"):
                validate_step({"claims":[{"line":1,"quote":"A"*301,"claim":"hidden suffix"}],
                               "next":None,"terminal":"unresolved"},ctx,"keras",sources)
            claims,_,_=validate_step({"claims":[{"line":1,"quote":"A"*40,"claim":"visible prefix"}],
                                      "next":None,"terminal":"unresolved"},ctx,"keras",sources)
            self.assertEqual(claims[0]["quote"],"A"*40)

    def test_source_end_requires_validated_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,_,_=self.fixture(root)
            ctx=context(sources,"keras","keras/layers/dense.py",1)
            empty={"claims":[],"next":None,"terminal":"source_end"}
            with self.assertRaisesRegex(ValueError,"source_end requires validated source claims"):
                validate_step(empty,ctx,"keras",sources)
            sourced={"claims":[{"line":1,"quote":"class Dense(Layer):","claim":"Dense class"}],
                     "next":None,"terminal":"source_end"}
            claims,next_step,terminal=validate_step(sourced,ctx,"keras",sources)
            self.assertEqual(claims[0]["quote"],"class Dense(Layer):")
            self.assertIsNone(next_step)
            self.assertEqual(terminal,"source_end")

    def test_routed_kilo_model_is_persisted_and_reused_for_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            routed="nvidia/nemotron-3-ultra-550b-a55b:free"
            worker_payloads=[]
            validator_payloads=[]
            async def model(stage,payload,prompt):
                if prompt=="planner": return {"focus":"trace source"}
                if prompt=="agent_trace_worker":
                    worker_payloads.append(payload.copy())
                    return {"claims":[{"line":1,"quote":"class Dense(Layer):","claim":"class"}],
                            "next":None,"terminal":"unresolved","_kilo_model":routed}
                validator_payloads.append(payload.copy())
                return {"approved_indices":[0],"_kilo_model":routed}
            first=asyncio.run(run(db,sources,1,1,model))
            self.assertEqual(first[0]["state"],"pending")
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute(
                    "SELECT model_id FROM trace_model_assignments WHERE layer_id='dense'"
                ).fetchone()[0],routed)
                conn.execute("UPDATE trace_tasks SET available_at=0 WHERE id='dense-root'")
            asyncio.run(run(db,sources,1,1,model))
            self.assertNotIn("_kilo_model",worker_payloads[0])
            self.assertEqual(worker_payloads[1]["_kilo_model"],routed)
            self.assertEqual(validator_payloads[0]["_kilo_model"],routed)

    def test_layer_inventory_and_reset_preserve_external_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); sources, db, _ = self.fixture(root)
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE sm75_kernel_index(id TEXT)")
                conn.execute("INSERT INTO sm75_kernel_index VALUES('keep')")
            reset(db)
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM sm75_kernel_index").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT count(*) FROM trace_layers").fetchone()[0], 0)
            self.assertTrue(db.with_name("cases.sqlite.before-trace-reset.bak").exists())

    def test_duplicate_layer_roots_are_aliased_without_deleting_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=root/"sources"
            source=sources/"keras"/"keras"/"layers"/"dense.py"
            source.parent.mkdir(parents=True)
            source.write_text("class Dense(Layer):\n    def call(self, x):\n        return x\n")
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            from tools.trace_orchestrator import stable
            canonical=stable("keras","keras/layers/dense.py",1,"Dense",digest)
            db=root/"layers.sqlite"
            with sqlite3.connect(db) as conn:
                setup(conn)
                for layer_id in (canonical,"legacy-duplicate"):
                    conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                                 (layer_id,"keras","Dense","keras/layers/dense.py",1,digest,"layer_class","pending"))
                    task_id="root-"+layer_id
                    conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                                 (task_id,layer_id,"keras/layers/dense.py",1,"{}"))
                conn.execute("INSERT INTO trace_results(task_id,layer_id,conditions,outcome,evidence) VALUES(?,?,?,?,?)",
                             ("root-legacy-duplicate","legacy-duplicate","{}","completed",json.dumps({"kept":True})))
            result=deduplicate_layer_inventory(db)
            self.assertEqual(result["aliases"],1)
            self.assertTrue(Path(result["backup"]).is_file())
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute(
                    "SELECT state FROM trace_layers WHERE id='legacy-duplicate'"
                ).fetchone()[0],f"duplicate_of:{canonical}")
                self.assertEqual(conn.execute(
                    "SELECT state FROM trace_tasks WHERE id='root-legacy-duplicate'"
                ).fetchone()[0],"suppressed")
                self.assertEqual(conn.execute(
                    "SELECT evidence FROM trace_results WHERE task_id='root-legacy-duplicate'"
                ).fetchone()[0],json.dumps({"kept":True}))
                enqueue_layers(conn,sources)
                self.assertEqual(conn.execute(
                    "SELECT count(*) FROM trace_layers WHERE state NOT LIKE 'duplicate_of:%'"
                ).fetchone()[0],1)
                self.assertEqual(conn.execute(
                    "SELECT count(*) FROM trace_layers"
                ).fetchone()[0],2)

    def test_duplicate_cases_keep_alias_payload_and_merge_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            with sqlite3.connect(db) as conn:
                seeded=seed_branch_cases(conn,sources)
                self.assertEqual(seeded["cases_inserted"],1)
                row=conn.execute("SELECT * FROM optimization_cases").fetchone()
                cols=[item[1] for item in conn.execute("PRAGMA table_info(optimization_cases)")]
                case=dict(zip(cols,row))
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("legacy","keras","Dense","keras/layers/dense.py",1,
                              hashlib.sha256(dense.read_bytes()).hexdigest(),"layer_class","duplicate_of:dense"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("legacy-root","legacy","keras/layers/dense.py",1,"{}"))
                alias=dict(case);alias["case_id"]="TRACE_legacy_duplicate"
                conn.execute(
                    "INSERT INTO optimization_cases ("+",".join(cols)+") VALUES ("+",".join("?" for _ in cols)+")",
                    tuple(alias[name] for name in cols))
                evidence=conn.execute(
                    "SELECT source_sha256,branch_line,evidence FROM trace_case_evidence WHERE case_id=?",
                    (case["case_id"],)).fetchone()
                conn.execute("INSERT INTO trace_case_evidence VALUES(?,?,?,?,?,?)",
                             (alias["case_id"],"legacy-root","legacy",evidence[0],evidence[1],evidence[2]))
                conn.execute("DELETE FROM trace_case_semantics")
            result=deduplicate_trace_cases(db)
            self.assertEqual(result["aliases"],1)
            self.assertTrue(Path(result["backup"]).is_file())
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute("SELECT case_id FROM optimization_cases").fetchall(),
                                 [(case["case_id"],)])
                self.assertEqual(conn.execute(
                    "SELECT canonical_case_id FROM trace_case_aliases WHERE alias_case_id=?",
                    (alias["case_id"],)).fetchone()[0],case["case_id"])
                self.assertEqual(conn.execute(
                    "SELECT count(*) FROM trace_case_evidence WHERE case_id=?",
                    (case["case_id"],)).fetchone()[0],2)
                new_alias=dict(case);new_alias["case_id"]="TRACE_new_duplicate"
                _persist_trace_case(conn,new_alias,"new-task","dense",evidence[0],evidence[1])
                self.assertEqual(conn.execute("SELECT count(*) FROM optimization_cases").fetchone()[0],1)
                self.assertEqual(conn.execute(
                    "SELECT canonical_case_id FROM trace_case_aliases WHERE alias_case_id='TRACE_new_duplicate'"
                ).fetchone()[0],case["case_id"])
                self.assertEqual(conn.execute(
                    "SELECT count(*) FROM trace_case_evidence WHERE case_id=?",
                    (case["case_id"],)).fetchone()[0],3)

    def test_inventory_skips_rescan_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            with sqlite3.connect(db) as conn:
                with patch("tools.trace_orchestrator.scan_layers", side_effect=AssertionError("rescan")):
                    self.assertEqual(ensure_layer_inventory(conn,sources),0)

    def test_scheduler_round_robins_between_layers_before_deeper_dfs_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            bn=sources/"pytorch"/"torch/nn/modules/batchnorm.py"
            bn.parent.mkdir(parents=True)
            bn.write_text("class BatchNorm2d:\n    def forward(self, x):\n        if self.training:\n            return x\n        return x\n")
            digest=hashlib.sha256(bn.read_bytes()).hexdigest()
            with sqlite3.connect(db) as conn:
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                    ("bn","pytorch","BatchNorm2d","torch/nn/modules/batchnorm.py",1,digest,"module_class","pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                    ("bn-root","bn","torch/nn/modules/batchnorm.py",1,"{}"))
            result=asyncio.run(run(db,sources,2,1,self.model))
            self.assertEqual({item["task"] for item in result},{"dense-root","bn-root"})
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM trace_scheduler").fetchone()[0],2)

    def test_branch_case_seeding_is_source_backed_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            with sqlite3.connect(db) as conn:
                first=seed_branch_cases(conn,sources)
                second=seed_branch_cases(conn,sources)
                case=conn.execute("SELECT applicability_predicate,evidence_status,validation_status FROM optimization_cases").fetchone()
                evidence=conn.execute("SELECT task_id,source_sha256,branch_line,evidence FROM trace_case_evidence").fetchone()
                total=conn.execute("SELECT count(*) FROM optimization_cases").fetchone()[0]
            self.assertEqual(first["candidates"],1)
            self.assertEqual(first["cases_inserted"],1)
            self.assertEqual(first["evidence_rows_added"],1)
            self.assertTrue(first["backup"])
            self.assertEqual(second["cases_inserted"],0)
            self.assertEqual(second["evidence_rows_added"],0)
            self.assertEqual(total,1)
            self.assertEqual(case,("self.use_bias","INFERRED","NEEDS_REVIEW"))
            self.assertEqual(evidence[0],"dense-root")
            self.assertEqual(evidence[1],hashlib.sha256(dense.read_bytes()).hexdigest())
            self.assertEqual(evidence[2],3)
            self.assertIn('"evidence_status": "INFERRED"',evidence[3])

    def test_training_mode_alone_is_not_an_optimization_branch(self):
        raw = (
            "def f(self, x, training):\n"
            "    if training:\n"
            "        return dropout(x)\n"
            "    if training and 0 < self.rate < 1:\n"
            "        return dropout(x)\n"
            "    return x\n"
        )
        self.assertEqual(branch_candidates(raw, 1), [(4, "training and 0 < self.rate < 1")])

    def test_case_pruning_refuses_to_run_during_a_trace_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources"
            db = root / "cases.sqlite"
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_leases VALUES(?,?,unixepoch()+600)",
                             ("active-task", "lease-token"))
                with self.assertRaisesRegex(RuntimeError, "live leases"):
                    prune_ineligible_optimization_cases(conn, sources)
                self.assertEqual(conn.execute(
                    "SELECT count(*) FROM optimization_case_exclusions"
                ).fetchone()[0], 0)

    def test_input_shape_and_backend_checks_require_optimization_control(self):
        raw = (
            "def f(x, rank, input_size):\n"
            "    if backend.ops.is_tensor(x):\n"
            "        return x + 1\n"
            "    if backend.backend() == \"tensorflow\":\n"
            "        return x + 2\n"
            "    if len(x.shape) == 4:\n"
            "        return x + 3\n"
            "    if rank == 3:\n"
            "        return x + 4\n"
            "    if x.dim() == 3:\n"
            "        return x + 5\n"
            "    if input_size % 8 == 0:\n"
            "        return x + 6\n"
            "    if use_tf32:\n"
            "        return x + 7\n"
            "    return x\n"
        )
        self.assertEqual(branch_candidates(raw, 1), [
            (12, "input_size % 8 == 0"),
            (14, "use_tf32"),
        ])

    def test_static_shape_gate_for_vectorized_implementation_is_eligible(self):
        raw = (
            "def f(inputs):\n"
            "    if inputs.shape[0] is not None:\n"
            "        return ops.stack(inputs)\n"
            "    return ops.vectorized_map(inputs)\n"
        )
        self.assertEqual(branch_candidates(raw, 1), [
            (2, "inputs.shape[0] is not None"),
        ])

    def test_static_shape_gate_using_constant_shape_path_is_eligible(self):
        raw = (
            "def f(inputs):\n"
            "    if inputs.shape.rank is not None:\n"
            "        return constant_op.constant(inputs.shape)\n"
            "    return dynamic_shape(inputs)\n"
        )
        self.assertEqual(branch_candidates(raw, 1), [
            (2, "inputs.shape.rank is not None"),
        ])

    def test_seed_archives_old_training_only_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources"
            source = sources / "keras" / "keras" / "layers" / "training_only.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "class TrainingOnly(Layer):\n"
                "    def call(self, x, training):\n"
                "        if training:\n"
                "            return x + 1\n"
                "        if self.use_bias:\n"
                "            return x + 2\n"
                "        return x\n",
                encoding="utf-8",
            )
            db = root / "cases.sqlite"
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("training-only", "keras", "TrainingOnly",
                              "keras/layers/training_only.py", 1, digest,
                              "layer_class", "pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("training-root", "training-only",
                              "keras/layers/training_only.py", 1, "{}"))
                ctx = context(sources, "keras", "keras/layers/training_only.py", 1)
                case = _trace_case(
                    conn,
                    ("training-root", "training-only", "keras", "keras/layers/training_only.py"),
                    ctx,
                    ("keras/layers/training_only.py:5", "self.use_bias"),
                )
                case["case_id"] = "TRACE_legacy_training_only"
                case["applicability_predicate"] = "training"
                case["optimization_name"] = "source branch: training"
                _persist_trace_case(conn, case, "training-root", "training-only", digest, 3)
                conn.commit()
                result = seed_branch_cases(conn, sources)
                active = conn.execute("SELECT applicability_predicate FROM optimization_cases").fetchall()
                archived = conn.execute(
                    "SELECT case_payload,evidence_payload,reason FROM optimization_case_exclusions"
                ).fetchone()
            self.assertEqual(result["cases_excluded"], 1)
            self.assertEqual(active, [("self.use_bias",)])
            self.assertTrue(Path(result["exclusion_backup"]).is_file())
            self.assertIn("training", archived[0])
            self.assertIn("training-root", archived[1])
            self.assertIn("no current source-backed optimization branch", archived[2])

    def test_python_branch_candidate_requires_optimization_dependency(self):
        raw = "def f(x, use_bias):\n    if x is None:\n        raise ValueError()\n    if use_bias:\n        return x + 1\n    return x\n"
        self.assertEqual(branch_candidates(raw,1),[(4,"use_bias")])

    def test_bias_branch_is_detected_from_class_root(self):
        lines=["class Linear(Module):","    def __init__(self, bias):","        if bias:",
               "            self.bias = Parameter()","        else:",
               "            self.register_parameter('bias', None)"]
        ctx={"file":"torch/nn/modules/linear.py","line":1,"lines":lines}
        self.assertEqual(next_unvisited_branch(ctx,{}),(3,"bias",1,6))

    def test_dfs_moves_to_next_class_method(self):
        lines=["class Linear(Module):","    def reset_parameters(self):","        if self.bias is not None:",
               "            self.bias.fill_(0)","    def forward(self, x):",
               "        return F.linear(x, self.weight, self.bias)"]
        ctx={"file":"torch/nn/modules/linear.py","line":3,"lines":lines}
        self.assertEqual(next_class_method(ctx,[]),(5,"class_method:5",1,6))

    def test_branch_fork_persists_both_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/"queue.sqlite"
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("l","pytorch","Linear","torch/nn/modules/linear.py",1,"h","module_class","pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("parent","l","torch/nn/modules/linear.py",2,"{}"))
                task=conn.execute("SELECT id,layer_id,parent_id,file,line,decisions,visited,depth FROM trace_tasks").fetchone()
                child,decisions=fork(conn,task,"torch/nn/modules/linear.py:2","bias",True)
                self.assertEqual(decisions,{"torch/nn/modules/linear.py:2":True})
                self.assertEqual(json.loads(conn.execute("SELECT decisions FROM trace_tasks WHERE id=?",(child,)).fetchone()[0]),
                                 {"torch/nn/modules/linear.py:2":False})

    def test_native_dispatch_branch_is_persisted_from_agent_dfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=root/"sources"
            src=sources/"pytorch"/"aten/src/ATen/native/Linear.cpp"
            src.parent.mkdir(parents=True)
            src.write_text(
                "Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {\n"
                "  const auto input_dim = input.dim();\n"
                "  if (input_dim == 2 && bias.defined()) {\n"
                "    return at::addmm(bias, input, weight.t());\n"
                "  }\n"
                "  return at::matmul(input, weight.t());\n"
                "}\n"
            )
            db=root/"queue.sqlite"
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("native","pytorch","Linear","aten/src/ATen/native/Linear.cpp",1,hashlib.sha256(src.read_bytes()).hexdigest(),"module_class","pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("root","native","aten/src/ATen/native/Linear.cpp",1,"{}"))
            async def model(stage,payload,prompt):
                if prompt=="planner": return {"focus":"inspect optimized dispatch"}
                if prompt=="agent_trace_worker":
                    line=payload["line"]
                    quote=next(v.split(": ",1)[1] for v in payload["source"].splitlines() if v.startswith(f"{line}: "))
                    return {"claims":[{"line":line,"quote":quote,"claim":"quoted native source"}],
                            "next":None,"terminal":"unresolved"}
                return {"approved_indices":[0]}
            asyncio.run(run(db,sources,2,1,model))
            with sqlite3.connect(db) as conn:
                case=conn.execute("SELECT source_file,source_function,applicability_predicate,evidence_status,validation_status FROM optimization_cases").fetchone()
                self.assertEqual(case,("aten/src/ATen/native/Linear.cpp","linear",
                                       "input_dim == 2 && bias.defined()","INFERRED","NEEDS_REVIEW"))
                self.assertEqual(conn.execute("SELECT count(*) FROM trace_case_evidence").fetchone()[0],1)
                self.assertEqual(conn.execute("SELECT count(*) FROM trace_tasks WHERE parent_id='root'").fetchone()[0],1)

    def test_native_shape_branch_requires_an_optimization_dispatch(self):
        adaptation = (
            "Tensor f(const Tensor& x) {\n"
            "  if (x.dim() == 3) {\n"
            "    return x.view({1, 2, 3});\n"
            "  }\n"
            "  return x.clone();\n"
            "}\n"
        )
        optimized = (
            "Tensor f(const Tensor& x, const Tensor& w) {\n"
            "  if (x.dim() == 2) {\n"
            "    return at::addmm(x, x, w);\n"
            "  }\n"
            "  return at::matmul(x, w);\n"
            "}\n"
        )
        self.assertEqual(native_branch_nodes(adaptation, 2), [])
        self.assertEqual(len(native_branch_nodes(optimized, 2)), 1)

    def test_verified_linear_binding_uses_quoted_source_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=Path(tmp)/"sources"; repo=sources/"pytorch"
            src=repo/"torch/nn/modules/linear.py"; target=repo/"torch/nn/functional.py"
            src.parent.mkdir(parents=True); target.parent.mkdir(parents=True,exist_ok=True)
            src.write_text("class Linear:\n    def forward(self, x):\n        return F.linear(x, self.weight, self.bias)\n")
            target.write_text("\n".join([""]*9+["        torch._C._nn.linear,"]))
            ctx=context(sources,"pytorch","torch/nn/modules/linear.py",3)
            claims=[{"quote":"return F.linear(x, self.weight, self.bias)"}]
            link=verified_linear_binding_target(sources,"pytorch","torch/nn/modules/linear.py",ctx,claims)
            self.assertEqual(link["file"],"torch/nn/functional.py")
            self.assertEqual(link["line"],10)
            self.assertIsNone(verified_linear_binding_target(sources,"pytorch","torch/nn/modules/linear.py",ctx,[]))

    def test_dfs_follows_inherited_batchnorm_forward(self):
        lines = [
            "class _BatchNorm(Module):",
            "    def forward(self, input):",
            "        return F.batch_norm(input, self.running_mean, self.running_var,",
            "                            self.weight, self.bias, self.training)",
            "class BatchNorm2d(_BatchNorm):",
            "    def _check_input_dim(self, input):",
            "        if input.dim() != 4:",
            "            raise ValueError()",
        ]
        ctx = {"line": 6, "lines": lines}
        self.assertEqual(
            next_class_method(ctx, []),
            (2, "class_method:2", 1, 4),
        )

    def test_linear_binding_resolves_python_alias_to_native_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=Path(tmp)/"sources"; repo=sources/"pytorch"
            module=repo/"torch/nn/modules/linear.py"
            functional=repo/"torch/nn/functional.py"
            yaml=repo/"aten/src/ATen/native/native_functions.yaml"
            native=repo/"aten/src/ATen/native/Linear.cpp"
            for path in (module,functional,yaml,native): path.parent.mkdir(parents=True,exist_ok=True)
            module.write_text("return F.linear(x, self.weight, self.bias)\n")
            functional.write_text("linear = _add_docstr(\n    torch._C._nn.linear,\n)\n")
            yaml.write_text("- func: linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor\n  dispatch:\n    CompositeImplicitAutograd: linear\n- func: linear_backward(...)\n")
            native.write_text("Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {\n  return input;\n}\n")
            ctx=context(sources,"pytorch","torch/nn/modules/linear.py",1)
            target=linear_binding_candidate(sources,"pytorch","torch/nn/modules/linear.py",ctx)
            self.assertEqual((target["file"],target["line"]),("torch/nn/functional.py",2))
            fctx=context(sources,"pytorch","torch/nn/functional.py",2)
            target=linear_binding_candidate(sources,"pytorch","torch/nn/functional.py",fctx)
            self.assertEqual((target["file"],target["line"]),("aten/src/ATen/native/native_functions.yaml",1))
            yctx=context(sources,"pytorch","aten/src/ATen/native/native_functions.yaml",1)
            target=linear_binding_candidate(sources,"pytorch","aten/src/ATen/native/native_functions.yaml",yctx)
            self.assertEqual((target["file"],target["line"]),("aten/src/ATen/native/Linear.cpp",1))

    def test_linear_addmm_resolves_to_cuda_dispatch_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=Path(tmp)/"sources"; repo=sources/"pytorch"
            native=repo/"aten/src/ATen/native/Linear.cpp"
            yaml=repo/"aten/src/ATen/native/native_functions.yaml"
            cuda=repo/"aten/src/ATen/native/cuda/Blas.cpp"
            for path in (native,yaml,cuda):
                path.parent.mkdir(parents=True,exist_ok=True)
            native.write_text("return at::addmm(bias, input, weight.t());\n")
            yaml.write_text(
                "- func: addmm.out(Tensor self, Tensor mat1, Tensor mat2) -> Tensor\n"
                "  dispatch:\n    CUDA: addmm_out_cuda\n"
                "- func: mm(Tensor self, Tensor mat2) -> Tensor\n")
            cuda.write_text(
                "TORCH_IMPL_FUNC(addmm_out_cuda)(const Tensor& self) {\n"
                "  addmm_out_cuda_impl(result, self, mat1, mat2, beta, alpha);\n"
                "}\n")
            ctx=context(sources,"pytorch","aten/src/ATen/native/Linear.cpp",1)
            candidate=linear_addmm_candidate(
                sources,"pytorch","aten/src/ATen/native/Linear.cpp",ctx)
            self.assertEqual(
                (candidate["file"],candidate["line"]),
                ("aten/src/ATen/native/native_functions.yaml",1))
            yctx=context(sources,"pytorch","aten/src/ATen/native/native_functions.yaml",1)
            candidate=linear_addmm_candidate(
                sources,"pytorch","aten/src/ATen/native/native_functions.yaml",yctx)
            self.assertEqual(
                (candidate["file"],candidate["line"]),
                ("aten/src/ATen/native/cuda/Blas.cpp",1))

    def test_cross_file_navigation_requires_shared_quoted_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=Path(tmp)/"sources"; current=sources/"pytorch"/"a.py"; target=sources/"pytorch"/"b.py"
            current.parent.mkdir(parents=True); current.write_text("return linear(x)\n")
            target.write_text("def linear(x):\n    return x\n")
            ctx=context(sources,"pytorch","a.py",1)
            good={"claims":[{"line":1,"quote":"linear(x)","claim":"calls linear"}],
                  "next":{"file":"b.py","line":1,"symbol":"linear","quote":"def linear(x):"},
                  "terminal":"unresolved"}
            self.assertEqual(validate_step(good,ctx,"pytorch",sources)[1]["file"],"b.py")
            bad={**good,"next":{**good["next"],"symbol":"fake"}}
            self.assertIsNone(validate_step(bad,ctx,"pytorch",sources)[1])

    def test_same_file_navigation_requires_exact_target_quote(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=Path(tmp)/"sources"; path=sources/"pytorch"/"x.py"
            path.parent.mkdir(parents=True); path.write_text("x=1\nreturn x\n")
            ctx=context(sources,"pytorch","x.py",1)
            response={"claims":[{"line":1,"quote":"x=1","claim":"assign"}],
                      "next":{"file":"x.py","line":2,"quote":"return x"},"terminal":"unresolved"}
            self.assertEqual(validate_step(response,ctx,"pytorch",sources)[1]["line"],2)

    def test_invalid_claim_error_includes_source_for_retry_feedback(self):
        ctx={"file":"linear.py","line":1,"lines":["return F.linear(x, weight, bias)"],
             "source":"1: return F.linear(x, weight, bias)","sha256":"hash","branches":[]}
        with self.assertRaisesRegex(ValueError,
                "claim lacks exact source evidence: line=1.*source='return F.linear"):
            validate_step({"claims":[{"line":1,"quote":"return F.linear(x, w, b)","claim":"bad quote"}],
                           "next":None,"terminal":"unresolved"},ctx,"pytorch",None)

    def test_navigation_rejects_test_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"sources"; p=root/"pytorch"/"tests"/"bad.py"
            p.parent.mkdir(parents=True);p.write_text("x=1\n")
            with self.assertRaises(ValueError):
                context(root,"pytorch","tests/bad.py",1)

    def test_sass_optimization_gate_requires_two_values_and_opposite_outcomes(self):
        edge={"optimization_dependency":{"dimension":"hyperparameter","name":"batch_size","observations":[
            {"value":1,"taken_count":2,"fallthrough_count":0},
            {"value":2,"taken_count":0,"fallthrough_count":3}]}}
        self.assertTrue(sass_optimization_dependent(edge))
        edge["optimization_dependency"]["observations"][1]["taken_count"]=1
        self.assertFalse(sass_optimization_dependent(edge))

    def test_sass_sequential_successor_stops_before_control_flow(self):
        lines=["/*0000*/ MOV R1, R2;","/*0010*/ IADD R1, R1, 1;","/*0020*/ @P0 BRA 0x40;","/*0030*/ EXIT;"]
        self.assertEqual(sass_sequential_successor(lines,1),{"line":3,"pc":32})
        self.assertIsNone(sass_sequential_successor(lines,3))

    def test_prune_ineligible_branch_retains_step_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            with sqlite3.connect(db) as conn:
                conn.execute("INSERT INTO trace_steps VALUES(?,?,?,?,?,?)",("dense-root",0,"x.py",1,"sha","{}"))
                conn.execute("UPDATE trace_tasks SET decisions=? WHERE id=?",
                             (json.dumps({"keras/layers/dense.py:2":True}),"dense-root"))
            result=prune_ineligible_branch_tasks(db,sources)
            self.assertEqual(result["marked_unresolved"],1)
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM trace_steps").fetchone()[0],1)

    def test_invalid_root_pruning_keeps_valid_root_and_backs_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            with sqlite3.connect(db) as conn:
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("bad","keras","Bad","keras/tests/test_bad.py",1,"x","layer_class","pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("bad-task","bad","keras/tests/test_bad.py",1,"{}"))
            result=prune_invalid_roots(db,sources)
            self.assertEqual(result["removed_roots"],1)
            self.assertTrue(Path(result["backup"]).is_file())

    def test_nemotron_supervises_worker_corrections_until_local_validation_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            calls=[]
            worker_responses=[
                {"claims":[{"line":1,"quote":"not source text","claim":"class"}],
                 "next":None,"terminal":"unresolved"},
                {"claims":[{"line":1,"quote":"False,","claim":"class declaration"}],
                 "next":None,"terminal":"unresolved"},
                {"claims":[{"line":1,"quote":"class Dense(Layer):","claim":"class declaration"}],
                 "next":None,"terminal":"unresolved"},
            ]
            async def model(stage,payload,prompt):
                calls.append((stage,prompt,payload.copy()))
                if prompt == "planner":
                    return {"focus":"follow source control flow"}
                if prompt == "agent_trace_worker":
                    if payload.get("supervisor_feedback"):
                        self.assertIn("claim lacks exact source evidence",payload["previous_failure"])
                        return worker_responses.pop(0)
                    return worker_responses.pop(0)
                if prompt == "agent_trace_supervisor_feedback":
                    self.assertIn("claim lacks exact source evidence",payload["validation_error"])
                    self.assertEqual(payload["repair_attempt_limit"],3)
                    if payload["repair_attempt"] == 2:
                        self.assertEqual(payload["correction_hints"],
                                         [{"line":1,"exact_visible_quote":"class Dense(Layer):"}])
                    return {"correction":"Copy the exact visible source line or omit that claim."}
                return {"approved_indices":[0]}
            result=asyncio.run(run(db,sources,1,1,model))
            self.assertEqual(result[0]["state"],"pending")
            self.assertEqual([call[1] for call in calls],
                             ["planner","agent_trace_worker","agent_trace_supervisor_feedback",
                              "agent_trace_worker","agent_trace_supervisor_feedback",
                              "agent_trace_worker","agent_trace_validator"])
            retry_calls=[call for call in calls if call[1]=="agent_trace_worker"]
            self.assertIn("Copy the exact visible source line",retry_calls[1][2]["supervisor_feedback"])
            self.assertIn("quote='False,'",calls[4][2]["validation_error"])
            with sqlite3.connect(db) as conn:
                step=json.loads(conn.execute("SELECT payload FROM trace_steps WHERE task_id='dense-root'").fetchone()[0])
                self.assertEqual(step["claims"][0]["quote"],"class Dense(Layer):")
                self.assertEqual(step["supervisor_model"],"openai/conductor")
                self.assertEqual(step["supervisor_feedback_attempts"],2)
                self.assertIsNone(step["repair_model"])
                self.assertEqual(step["repair_attempts"],0)

    def test_nemotron_supervises_claims_rejected_by_independent_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            calls=[]
            async def model(stage,payload,prompt):
                calls.append((prompt,payload.copy()))
                if prompt=="planner":
                    return {"focus":"follow source control flow"}
                if prompt=="agent_trace_worker":
                    if payload.get("supervisor_feedback"):
                        self.assertIn("unsupported claim",payload["supervisor_feedback"])
                        return {"claims":[{"line":1,"quote":"class Dense(Layer):",
                                          "claim":"Dense class declaration"}],
                                "next":None,"terminal":"source_end"}
                    return {"claims":[
                                {"line":1,"quote":"class Dense(Layer):",
                                 "claim":"claim is not supported by this source"},
                                {"line":1,"quote":"class Dense(Layer):",
                                 "claim":"Dense class declaration"}],
                            "next":None,"terminal":"source_end"}
                if prompt=="agent_trace_claim_feedback":
                    rejected=payload["rejected_claims"]
                    self.assertEqual([item["index"] for item in rejected],[0])
                    self.assertIn("does not support the claim",rejected[0]["reason"])
                    return {"correction":"Remove the unsupported claim and retain only the source-backed class declaration."}
                if prompt=="agent_trace_validator":
                    if len(payload["claims"])==2:
                        return {"approved_indices":[1],"rejected_indices":[0],
                                "rejection_reasons":[{"index":0,"reason":"Source does not support the claim."}]}
                    return {"approved_indices":[0],"rejected_indices":[],"rejection_reasons":[]}
                return {"next":None}

            result=asyncio.run(run(db,sources,1,1,model))
            self.assertEqual(result[0]["state"],"pending")
            self.assertEqual([name for name,_ in calls],
                             ["planner","agent_trace_worker","agent_trace_validator",
                              "agent_trace_claim_feedback","agent_trace_worker",
                              "agent_trace_validator"],result)
            with sqlite3.connect(db) as conn:
                step=json.loads(conn.execute(
                    "SELECT payload FROM trace_steps WHERE task_id='dense-root'"
                ).fetchone()[0])
                self.assertEqual(step["claims"][0]["claim"],"Dense class declaration")
                self.assertEqual(step["claim_feedback_attempts"],1)
                self.assertEqual(step["claim_review_history"][0]["rejected_claims"][0]["reason"],
                                 "Source does not support the claim.")
                self.assertEqual(step["claim_review_history"][-1]["approved_indices"],[0])

    def test_nemotron_recovers_a_failed_openkilo_worker_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            calls=[]
            async def model(stage,payload,prompt):
                calls.append((stage,prompt,payload.copy()))
                if prompt=="planner":
                    return {"focus":"follow source control flow"}
                if prompt=="agent_trace_worker":
                    raise RuntimeError("temporary OpenKilo transport failure")
                if prompt=="agent_trace_repair":
                    self.assertIsNone(payload["worker_response"])
                    self.assertIn("temporary OpenKilo transport failure",payload["worker_error"])
                    self.assertIn("class Dense(Layer):",payload["task"]["source"])
                    return {"claims":[{"line":1,"quote":"class Dense(Layer):","claim":"Dense class"}],
                            "next":None,"terminal":"unresolved"}
                return {"approved_indices":[0]}
            result=asyncio.run(run(db,sources,1,1,model))
            self.assertEqual(result[0]["state"],"pending")
            self.assertEqual([call[1] for call in calls],
                             ["planner","agent_trace_worker","agent_trace_repair","agent_trace_validator"])
            with sqlite3.connect(db) as conn:
                step=json.loads(conn.execute(
                    "SELECT payload FROM trace_steps WHERE task_id='dense-root'"
                ).fetchone()[0])
                self.assertEqual(step["repair_attempts"],1)
                self.assertEqual(step["repair_model"],"openai/conductor")
                self.assertIsNone(step["worker_model"])
                self.assertEqual(conn.execute("SELECT count(*) FROM trace_task_errors").fetchone()[0],0)

    def test_nemotron_retries_a_malformed_repair_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            repair_calls=[]
            async def model(stage,payload,prompt):
                if prompt=="planner":
                    return {"focus":"follow source control flow"}
                if prompt=="agent_trace_worker":
                    return {"claims":[{"line":1,"quote":"not source text","claim":"class"}],
                            "next":None,"terminal":"unresolved"}
                if prompt=="agent_trace_supervisor_feedback":
                    return {"correction":"Use an exact source quote."}
                if prompt=="agent_trace_repair":
                    repair_calls.append(payload.copy())
                    if payload["repair_attempt"]==1:
                        raise ValueError("trace repair response claims must be a list")
                    self.assertEqual(payload["repair_attempt"],2)
                    self.assertIn("claims must be a list",payload["validation_error"])
                    self.assertEqual(payload["worker_response"]["claims"][0]["quote"],"not source text")
                    return {"claims":[{"line":1,"quote":"class Dense(Layer):","claim":"class"}],
                            "next":None,"terminal":"unresolved"}
                return {"approved_indices":[0]}
            result=asyncio.run(run(db,sources,1,1,model))
            self.assertEqual(result[0]["state"],"pending")
            self.assertEqual([item["repair_attempt"] for item in repair_calls],[1,2])
            with sqlite3.connect(db) as conn:
                step=json.loads(conn.execute(
                    "SELECT payload FROM trace_steps WHERE task_id='dense-root'"
                ).fetchone()[0])
                self.assertEqual(step["repair_attempts"],2)
                self.assertEqual(conn.execute(
                    "SELECT count(*) FROM trace_task_errors"
                ).fetchone()[0],0)

    def test_retry_unresolved_validation_errors_backs_up_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE trace_tasks SET state='unresolved',attempts=2,error=?,result=? WHERE id=?",
                    ("worker: ValueError: claim lacks exact source evidence: quote invalid",
                     "old task result", "dense-root"),
                )
                conn.execute(
                    "INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth,state) VALUES(?,?,?,?,?,0,?)",
                    ("source-end", "dense", "keras/layers/dense.py", 3, "{}", "unresolved"),
                )
                conn.execute(
                    "INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth,state,attempts,error) VALUES(?,?,?,?,?,0,?,?,?)",
                    ("validator-bad", "dense", "keras/layers/dense.py", 1, "{}", "unresolved", 1,
                     "worker: ValueError: trace validator response has invalid approved_indices"),
                )
                conn.execute(
                    "INSERT INTO trace_task_errors(task_id,attempt,stage,error) VALUES(?,?,?,?)",
                    ("dense-root", 2, "worker", "historical validation failure"),
                )
                conn.execute("INSERT INTO trace_results(task_id,layer_id,conditions,outcome,evidence) VALUES(?,?,?,?,?)",
                             ("dense-root", "dense", "{}", "unresolved", '{"reason":"old"}'))
            result=retry_unresolved_validation_errors(db)
            self.assertEqual(result["requeued"],2)
            self.assertTrue(Path(result["backup"]).is_file())
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute(
                    "SELECT state,attempts,available_at,error,result,retry_priority FROM trace_tasks WHERE id='dense-root'"
                ).fetchone(), ("pending",2,0,None,None,1))
                self.assertEqual(conn.execute(
                    "SELECT state,error FROM trace_tasks WHERE id='source-end'"
                ).fetchone(), ("unresolved",None))
                self.assertEqual(conn.execute(
                    "SELECT state,attempts,error,retry_priority FROM trace_tasks WHERE id='validator-bad'"
                ).fetchone(), ("pending",1,None,1))
                self.assertEqual(conn.execute(
                    "SELECT error FROM trace_task_errors WHERE task_id='dense-root'"
                ).fetchone()[0], "historical validation failure")
                self.assertEqual(conn.execute(
                    "SELECT evidence FROM trace_results WHERE task_id='dense-root'"
                ).fetchone()[0], '{"reason":"old"}')
            with sqlite3.connect(result["backup"]) as backup:
                self.assertEqual(backup.execute(
                    "SELECT state,error FROM trace_tasks WHERE id='dense-root'"
                ).fetchone(), ("unresolved", "worker: ValueError: claim lacks exact source evidence: quote invalid"))

    def test_validation_recovery_priority_precedes_regular_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            with sqlite3.connect(db) as conn:
                digest=hashlib.sha256(dense.read_bytes()).hexdigest()
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("fresh-layer","keras","Fresh","keras/layers/dense.py",1,
                              digest,"layer_class","pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("fresh-task","fresh-layer","keras/layers/dense.py",1,"{}"))
                conn.execute("UPDATE trace_tasks SET attempts=1,error='previous failure',retry_priority=1 WHERE id='dense-root'")
                conn.execute("UPDATE trace_tasks SET attempts=2,error='repeated failure' WHERE id='fresh-task'")
                conn.execute("INSERT INTO trace_scheduler VALUES('dense',100)")
                conn.execute("INSERT INTO trace_scheduler VALUES('fresh-layer',0)")
            result=asyncio.run(run(db,sources,1,1,self.model,max_attempts=3))
            self.assertEqual(result[0]["task"],"dense-root")

    def test_failed_step_is_retried_with_previous_error_for_self_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            attempts={"planner":0}
            observed=[]
            async def model(stage,payload,prompt):
                if prompt=="planner":
                    attempts["planner"]+=1
                    if attempts["planner"]==1:
                        raise RuntimeError("temporary planner outage")
                    observed.append(payload["task"]["previous_failure"])
                    return {"focus":"recover from the recorded failure"}
                if prompt=="agent_trace_worker":
                    return {"claims":[{"line":1,"quote":"class Dense(Layer):","claim":"layer declaration"}],
                            "next":None,"terminal":"unresolved"}
                return {"approved_indices":[0]}
            first=asyncio.run(run(db,sources,1,1,model,max_attempts=2))
            self.assertEqual(first[0]["state"],"pending")
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute(
                    "SELECT stage,error FROM trace_task_errors"
                ).fetchone(),("planner","planner: RuntimeError: temporary planner outage"))
                conn.execute("UPDATE trace_tasks SET available_at=0,retry_priority=1 WHERE id='dense-root'")
            second=asyncio.run(run(db,sources,1,1,model,max_attempts=2))
            self.assertEqual(second[0]["state"],"pending")
            self.assertIn("temporary planner outage",observed[0])
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute(
                    "SELECT state,attempts,error,line,retry_priority FROM trace_tasks WHERE id='dense-root'"
                ).fetchone(),("pending",0,None,3,0))

    def test_repeated_invalid_worker_output_is_saved_unresolved_after_correction_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            worker_attempts=[]
            async def bad_quote(stage,payload,prompt):
                if prompt == "planner":
                    return {"focus":"follow source control flow"}
                if prompt == "agent_trace_worker":
                    worker_attempts.append(payload["previous_failure"])
                    return {"claims":[{"line":1,"quote":"not in source","claim":"bad"}],
                            "next":None,"terminal":"unresolved"}
                if prompt == "agent_trace_supervisor_feedback":
                    return {"correction":"Copy an exact quote or omit the claim."}
                if prompt == "agent_trace_repair":
                    return {"claims":[{"line":1,"quote":"not in source","claim":"bad"}],
                            "next":None,"terminal":"unresolved"}
                return {"approved_indices":[]}
            first=asyncio.run(run(db,sources,1,1,bad_quote,max_attempts=4))
            self.assertEqual(first[0]["state"],"pending")
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE trace_tasks SET available_at=0 WHERE id='dense-root'")
            second=asyncio.run(run(db,sources,1,1,bad_quote,max_attempts=4))
            self.assertEqual(second[0]["state"],"unresolved")
            self.assertIsNone(worker_attempts[0])
            self.assertIn("claim lacks exact source evidence",worker_attempts[1])
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute(
                    "SELECT state,attempts FROM trace_tasks WHERE id='dense-root'"
                ).fetchone(),("unresolved",1))
                saved=json.loads(conn.execute(
                    "SELECT evidence FROM trace_results WHERE task_id='dense-root'"
                ).fetchone()[0])
                self.assertIn("claim lacks exact source evidence",saved["reason"])
                self.assertEqual(conn.execute(
                    "SELECT count(*) FROM trace_task_errors WHERE task_id='dense-root'"
                ).fetchone()[0],2)

    def test_repeated_invalid_next_location_is_saved_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            async def invalid_target(stage,payload,prompt):
                if prompt == "planner":
                    return {"focus":"follow source control flow"}
                if prompt == "agent_trace_worker":
                    return {"claims":[{"line":1,"quote":"class Dense(Layer):","claim":"class"}],
                            "next":{"file":"keras/layers/dense.py","line":"999",
                                    "quote":"missing"},"terminal":None}
                if prompt == "agent_trace_supervisor_feedback":
                    return {"correction":"Use a valid source target or return unresolved."}
                if prompt == "agent_trace_repair":
                    return {"claims":[{"line":1,"quote":"class Dense(Layer):","claim":"class"}],
                            "next":{"file":"keras/layers/dense.py","line":"999",
                                    "quote":"missing"},"terminal":None}
                return {"approved_indices":[0]}
            asyncio.run(run(db,sources,1,1,invalid_target,max_attempts=4))
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE trace_tasks SET available_at=0 WHERE id='dense-root'")
            second=asyncio.run(run(db,sources,1,1,invalid_target,max_attempts=4))
            self.assertEqual(second[0]["state"],"unresolved")
            with sqlite3.connect(db) as conn:
                saved=json.loads(conn.execute(
                    "SELECT evidence FROM trace_results WHERE task_id='dense-root'"
                ).fetchone()[0])
                self.assertIn("invalid next location",saved["reason"])

    def test_quota_wait_expiry_preserves_attempts_and_requeues_task(self):
        from optimization_explorer.llm import OpenKiloQuotaWaitTimeout
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE trace_tasks SET attempts=2 WHERE id='dense-root'")
            async def quota_wait(stage,payload,prompt):
                raise OpenKiloQuotaWaitTimeout("OpenKilo remained rate-limited")
            with patch.dict("os.environ", {"LLM_QUOTA_WAIT_SECONDS":"1200"}):
                result=asyncio.run(run(db,sources,1,1,quota_wait,max_attempts=3))
            self.assertEqual(result[0]["state"],"pending")
            self.assertEqual(result[0]["retry_in_seconds"],1200)
            with sqlite3.connect(db) as conn:
                state,attempts,available,error=conn.execute(
                    "SELECT state,attempts,available_at,error FROM trace_tasks WHERE id='dense-root'"
                ).fetchone()
                self.assertEqual((state,attempts),("pending",2))
                self.assertGreater(available,int(__import__('time').time())+1100)
                self.assertIn("OpenKilo remained rate-limited",error)

    def test_failed_step_is_left_marked_after_retry_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            async def broken(stage,payload,prompt):
                raise RuntimeError("persistent provider failure")
            result=asyncio.run(run(db,sources,1,1,broken,max_attempts=1))
            self.assertEqual(result[0]["state"],"failed")
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute(
                    "SELECT state,attempts,error FROM trace_tasks WHERE id='dense-root'"
                ).fetchone(),("failed",1,"planner: RuntimeError: persistent provider failure"))
                self.assertEqual(conn.execute(
                    "SELECT count(*) FROM trace_task_errors"
                ).fetchone()[0],1)

    def test_imported_base_class_navigation_candidate_is_source_backed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=root/"sources"
            current=sources/"keras"/"keras"/"layers"/"current.py"
            base=sources/"keras"/"keras"/"layers"/"base.py"
            current.parent.mkdir(parents=True)
            current.write_text("from keras.layers.base import Base\n\nclass Current(Base):\n    def __init__(self, x):\n        super().__init__(x)\n")
            base.write_text("class Base:\n    def __init__(self, x):\n        self.x = x\n")
            ctx=context(sources,"keras","keras/layers/current.py",5)
            candidate,claim=imported_base_class_target(sources,"keras",ctx)
            claims,validated,_=validate_step(
                {"claims":[claim],"next":candidate,"terminal":"unresolved"},
                ctx,"keras",sources,
            )
            self.assertEqual(candidate["file"],"keras/layers/base.py")
            self.assertEqual(candidate["line"],1)
            self.assertEqual(candidate["symbol"],"Base")
            self.assertEqual(claims[0]["quote"],"class Current(Base):")
            self.assertEqual(validated["quote"],"class Base:")

    def test_imported_module_alias_resolves_tensorflow_reexported_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=Path(tmp)/"sources"
            current=sources/"tensorflow"/"tensorflow/python/keras/legacy_tf_layers/pooling.py"
            package=sources/"tensorflow"/"tensorflow/python/keras/layers/__init__.py"
            implementation=sources/"tensorflow"/"tensorflow/python/keras/layers/pooling.py"
            current.parent.mkdir(parents=True)
            package.parent.mkdir(parents=True)
            implementation.parent.mkdir(parents=True,exist_ok=True)
            current.write_text(
                "from tensorflow.python.keras import layers as keras_layers\n"
                "from tensorflow.python.keras.legacy_tf_layers import base\n\n"
                "class MaxPooling2D(keras_layers.MaxPooling2D, base.Layer):\n"
                "  def __init__(self):\n"
                "    super(MaxPooling2D, self).__init__()\n")
            package.write_text(
                "from tensorflow.python.keras.layers.pooling import MaxPooling2D\n")
            implementation.write_text(
                "class MaxPooling2D(Pooling2D):\n"
                "  def __init__(self):\n"
                "    super(MaxPooling2D, self).__init__()\n")
            ctx=context(sources,"tensorflow",
                        "tensorflow/python/keras/legacy_tf_layers/pooling.py",6)
            candidate,claim=imported_base_class_target(sources,"tensorflow",ctx)
            self.assertEqual(candidate["file"],
                             "tensorflow/python/keras/layers/pooling.py")
            self.assertEqual(candidate["symbol"],"MaxPooling2D")
            self.assertEqual(candidate["navigation_basis"],
                             "imported_base_class_candidate")
            self.assertEqual(claim["quote"],
                             "class MaxPooling2D(keras_layers.MaxPooling2D, base.Layer):")

    def test_nemotron_null_review_can_follow_an_imported_base_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources_root=root/"sources"
            current=sources_root/"keras"/"keras"/"layers"/"current.py"
            base=sources_root/"keras"/"keras"/"layers"/"base.py"
            current.parent.mkdir(parents=True)
            current.write_text("from keras.layers.base import Base\n\nclass Current(Base):\n    def __init__(self, x):\n        super().__init__(x)\n")
            base.write_text("class Base:\n    def __init__(self, x):\n        self.x = x\n")
            db=root/"cases.sqlite"
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("ops","keras","Current","keras/layers/current.py",3,
                              hashlib.sha256(current.read_bytes()).hexdigest(),"layer_class","pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("base-root","ops","keras/layers/current.py",5,"{}" ))
            async def model(stage,payload,prompt):
                if prompt == "planner": return {"focus":"trace base class"}
                if prompt == "agent_trace_worker":
                    return {"claims":[{"line":5,"quote":"super().__init__(x)","claim":"delegates to base initializer"}],
                            "next":None,"terminal":"unresolved"}
                if prompt == "agent_trace_navigation_review":
                    self.assertEqual(payload["navigation_candidate"]["symbol"],"Base")
                    return {"next":None,"terminal":"unresolved"}
                if prompt == "agent_trace_validator": return {"approved_indices":[0,1]}
                raise AssertionError(prompt)
            result=asyncio.run(run(db,sources_root,1,1,model))
            self.assertEqual(result[0]["state"],"pending")
            with sqlite3.connect(db) as conn:
                step=json.loads(conn.execute(
                    "SELECT payload FROM trace_steps WHERE task_id='base-root'"
                ).fetchone()[0])
            self.assertTrue(step["navigation_review_attempted"])
            self.assertEqual(step["next"]["file"],"keras/layers/base.py")
            self.assertEqual(step["next"]["line"],1)
            self.assertEqual(step["next"]["navigation_basis"],"imported_base_class_candidate")
            self.assertIn("Base",step["claims"][-1]["quote"])

    def test_empty_unresolved_response_with_local_call_gets_nemotron_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=root/"sources"
            current=sources/"keras"/"keras"/"layers"/"current.py"
            current.parent.mkdir(parents=True)
            current.write_text(
                "class Current:\n"
                "    def call(self, x):\n"
                "        return self._helper(x)\n"
                "    def _helper(self, x):\n"
                "        return x\n")
            db=root/"cases.sqlite"
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("ops","keras","Current","keras/layers/current.py",1,
                              hashlib.sha256(current.read_bytes()).hexdigest(),"layer_class","pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("empty-claims","ops","keras/layers/current.py",2,"{}"))
            calls=[]
            async def model(stage,payload,prompt):
                calls.append((stage,prompt,payload))
                if prompt == "planner":
                    return {"focus":"follow the exact source call"}
                if prompt == "agent_trace_worker" and not payload.get("supervisor_feedback"):
                    return {"claims":[],"next":None,"terminal":"unresolved"}
                if prompt == "agent_trace_supervisor_feedback":
                    return {"correction":"Quote the visible call and follow its local method."}
                if prompt == "agent_trace_worker":
                    return {"claims":[{"line":3,"quote":"return self._helper(x)",
                                        "claim":"calls the local helper"}],
                            "next":{"file":"keras/layers/current.py","line":4,
                                    "quote":"def _helper(self, x):"},
                            "terminal":"unresolved"}
                if prompt == "agent_trace_validator":
                    return {"approved_indices":[0]}
                raise AssertionError(prompt)
            result=asyncio.run(run(db,sources,1,1,model))
            self.assertEqual(result[0]["state"],"pending")
            feedback=next(payload for _,prompt,payload in calls
                          if prompt == "agent_trace_supervisor_feedback")
            self.assertIn("no source claims",feedback["validation_error"])
            with sqlite3.connect(db) as conn:
                step=json.loads(conn.execute(
                    "SELECT payload FROM trace_steps WHERE task_id='empty-claims'"
                ).fetchone()[0])
            self.assertEqual(step["supervisor_feedback_attempts"],1)
            self.assertEqual(step["next"]["line"],4)

    def test_unresolved_imported_base_navigation_retry_is_verified_and_backed_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=root/"sources"
            current=sources/"keras"/"keras"/"layers"/"current.py"
            base=sources/"keras"/"keras"/"layers"/"base.py"
            current.parent.mkdir(parents=True)
            current.write_text(
                "from keras.layers.base import Base\n\n"
                "class Current(Base):\n"
                "    def __init__(self, x):\n"
                "        super().__init__(x)\n")
            base.write_text("class Base:\n    def __init__(self, x):\n        self.x = x\n")
            db=root/"cases.sqlite"
            marker='keras/layers/current.py:5:{}'
            payload={"claims":[{"line":5,"quote":"super().__init__(x)",
                                "claim":"delegates to base initializer"}],
                     "next":None,"terminal":"unresolved"}
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("ops","keras","Current","keras/layers/current.py",3,
                              hashlib.sha256(current.read_bytes()).hexdigest(),"layer_class","pending"))
                conn.execute("""INSERT INTO trace_tasks
                    (id,layer_id,file,line,decisions,visited,depth,state,attempts,result)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    ("base-root","ops","keras/layers/current.py",5,"{}",
                     json.dumps(["earlier",marker]),2,"unresolved",1,"old result"))
                conn.execute("INSERT INTO trace_steps VALUES(?,?,?,?,?,?)",
                             ("base-root",2,"keras/layers/current.py",5,
                              hashlib.sha256(current.read_bytes()).hexdigest(),json.dumps(payload)))

            result=retry_unresolved_navigation_candidates(db,sources)
            self.assertEqual(result["requeued"],1)
            self.assertTrue(Path(result["backup"]).is_file())
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute(
                    "SELECT state,depth,visited,attempts,error,result,retry_priority "
                    "FROM trace_tasks WHERE id='base-root'"
                ).fetchone(),("pending",3,'["earlier"]',1,None,None,1))
                self.assertEqual(conn.execute(
                    "SELECT count(*) FROM trace_steps WHERE task_id='base-root'"
                ).fetchone()[0],1)
            with sqlite3.connect(result["backup"]) as backup_db:
                self.assertEqual(backup_db.execute(
                    "SELECT state,depth FROM trace_tasks WHERE id='base-root'"
                ).fetchone(),("unresolved",2))

    def test_same_line_call_resolves_to_local_function_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=Path(tmp)/"sources"
            source=sources/"keras"/"keras/src/ops/nn.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "def apply(x):\n"
                "    return _normalize(x)\n\n"
                "def _normalize(x):\n"
                "    return x\n")
            ctx=context(sources,"keras","keras/src/ops/nn.py",2)
            candidate,claim=same_file_call_target(ctx)
            self.assertEqual(candidate["line"],4)
            self.assertEqual(candidate["symbol"],"_normalize")
            self.assertEqual(candidate["navigation_basis"],"same_file_call_candidate")
            claims,validated,_=validate_step(
                {"claims":[claim],"next":candidate,"terminal":"unresolved"},
                ctx,"keras",sources,
            )
            self.assertEqual(claims[0]["quote"],"    return _normalize(x)")
            self.assertEqual(validated["line"],4)

    def test_nemotron_null_review_uses_exact_same_function_call_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources_root=root/"sources"
            source=sources_root/"keras"/"keras"/"layers"/"ops.py"
            source.parent.mkdir(parents=True)
            source.write_text("def apply(x):\n    return helper(x)\n")
            db=root/"cases.sqlite"
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("ops","keras","apply","keras/layers/ops.py",1,
                              hashlib.sha256(source.read_bytes()).hexdigest(),"functional_operation","pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("ops-root","ops","keras/layers/ops.py",1,"{}" ))
            async def model(stage,payload,prompt):
                if prompt == "planner": return {"focus":"trace source"}
                if prompt == "agent_trace_worker":
                    return {"claims":[{"line":2,"quote":"return helper(x)","claim":"call"}],
                            "next":None,"terminal":"unresolved"}
                if prompt == "agent_trace_navigation_review":
                    self.assertEqual(payload["navigation_candidate"]["line"],2)
                    return {"next":None,"terminal":"unresolved"}
                if prompt == "agent_trace_validator": return {"approved_indices":[0]}
                raise AssertionError(prompt)
            result=asyncio.run(run(db,sources_root,1,1,model))
            self.assertEqual(result[0]["state"],"pending")
            with sqlite3.connect(db) as conn:
                step=json.loads(conn.execute(
                    "SELECT payload FROM trace_steps WHERE task_id='ops-root'"
                ).fetchone()[0])
            self.assertEqual(step["next"]["line"],2)
            self.assertEqual(step["next"]["navigation_basis"],"next_source_call_candidate")
            self.assertTrue(step["navigation_review_attempted"])

    def test_nemotron_navigation_review_corrects_out_of_root_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources"
            current = sources / "keras" / "keras" / "layers" / "ops.py"
            current.parent.mkdir(parents=True)
            current.write_text("def apply(x):\n    return helper(x)\n")
            helper = sources / "keras" / "keras" / "layers" / "helper.py"
            helper.write_text("def helper(x):\n    return x\n")
            db = root / "cases.sqlite"
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("ops", "keras", "apply", "keras/layers/ops.py", 1,
                              hashlib.sha256(current.read_bytes()).hexdigest(),
                              "functional_operation", "pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("ops-root", "ops", "keras/layers/ops.py", 1, "{}"))
            navigation_calls = []

            async def model(stage, payload, prompt):
                if prompt == "planner":
                    return {"focus": "trace source continuation"}
                if prompt == "agent_trace_worker":
                    return {"claims": [{"line": 2, "quote": "return helper(x)", "claim": "call"}],
                            "next": None, "terminal": "unresolved"}
                if prompt == "agent_trace_navigation_review":
                    navigation_calls.append(payload.copy())
                    if len(navigation_calls) == 1:
                        return {"next": {"file": "../outside.py", "line": 1,
                                         "symbol": "helper", "quote": "def helper(x):"},
                                "terminal": "unresolved"}
                    self.assertIn("source outside eligible framework code", payload["validation_error"])
                    return {"next": {"file": "keras/layers/helper.py", "line": 1,
                                     "symbol": "helper", "quote": "def helper(x):"},
                            "terminal": "unresolved"}
                if prompt == "agent_trace_validator":
                    return {"approved_indices": [0]}
                raise AssertionError(prompt)

            result = asyncio.run(run(db, sources, 1, 1, model))
            self.assertEqual(result[0]["state"], "pending")
            self.assertEqual(len(navigation_calls), 2)
            with sqlite3.connect(db) as conn:
                step = json.loads(conn.execute(
                    "SELECT payload FROM trace_steps WHERE task_id='ops-root'"
                ).fetchone()[0])
            self.assertEqual(step["next"]["file"], "keras/layers/helper.py")
            self.assertEqual(step["navigation_feedback_attempts"], 1)
            self.assertIsNone(step["navigation_validation_error"])

    def test_nemotron_navigation_review_can_recover_a_verified_cross_file_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources_root=root/"sources"
            source=sources_root/"keras"/"keras"/"layers"/"ops.py"
            source.parent.mkdir(parents=True)
            source.write_text("def apply(x):\n    return helper(x)\n")
            helper=sources_root/"keras"/"keras"/"layers"/"helper.py"
            helper.write_text("def helper(x):\n    return x\n")
            db=root/"cases.sqlite"
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("ops","keras","apply","keras/layers/ops.py",1,
                              hashlib.sha256(source.read_bytes()).hexdigest(),"functional_operation","pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("ops-root","ops","keras/layers/ops.py",1,"{}"))
            reviewed=[]
            async def model(stage,payload,prompt):
                if prompt == "planner": return {"focus":"trace source"}
                if prompt == "agent_trace_worker":
                    return {"claims":[{"line":2,"quote":"return helper(x)","claim":"call"}],
                            "next":None,"terminal":"unresolved"}
                if prompt == "agent_trace_navigation_review":
                    reviewed.append(payload)
                    return {"next":{"file":"keras/layers/helper.py","line":1,
                                    "symbol":"helper","quote":"def helper(x):"},
                            "terminal":"unresolved"}
                if prompt == "agent_trace_validator": return {"approved_indices":[0]}
                raise AssertionError(prompt)
            result=asyncio.run(run(db,sources_root,1,1,model))
            self.assertEqual(result[0]["state"],"pending", result)
            self.assertEqual(len(reviewed),1)
            with sqlite3.connect(db) as conn:
                step=json.loads(conn.execute(
                    "SELECT payload FROM trace_steps WHERE task_id='ops-root'"
                ).fetchone()[0])
            self.assertEqual(step["next"]["file"],"keras/layers/helper.py")
            self.assertNotIn("navigation_basis", step["next"])
            self.assertTrue(step["navigation_review_attempted"])

    def test_nemotron_navigation_review_cannot_enqueue_an_unquoted_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources_root=root/"sources"
            source=sources_root/"keras"/"keras"/"layers"/"ops.py"
            source.parent.mkdir(parents=True)
            source.write_text("def apply(x):\n    return helper(x)\n")
            helper=sources_root/"keras"/"keras"/"layers"/"helper.py"
            helper.write_text("def helper(x):\n    return x\n")
            db=root/"cases.sqlite"
            with sqlite3.connect(db) as conn:
                setup(conn)
                conn.execute("INSERT INTO trace_layers VALUES(?,?,?,?,?,?,?,?)",
                             ("ops","keras","apply","keras/layers/ops.py",1,
                              hashlib.sha256(source.read_bytes()).hexdigest(),"functional_operation","pending"))
                conn.execute("INSERT INTO trace_tasks(id,layer_id,file,line,decisions,depth) VALUES(?,?,?,?,?,0)",
                             ("ops-root","ops","keras/layers/ops.py",1,"{}"))
            async def model(stage,payload,prompt):
                if prompt == "planner": return {"focus":"trace source"}
                if prompt == "agent_trace_worker":
                    return {"claims":[{"line":2,"quote":"return helper(x)","claim":"call"}],
                            "next":None,"terminal":"unresolved"}
                if prompt == "agent_trace_navigation_review":
                    return {"next":{"file":"keras/layers/helper.py","line":1,
                                    "symbol":"helper","quote":"fabricated target"},
                            "terminal":"unresolved"}
                if prompt == "agent_trace_validator": return {"approved_indices":[0]}
                raise AssertionError(prompt)
            result=asyncio.run(run(db,sources_root,1,1,model))
            self.assertEqual(result[0]["state"],"pending")
            with sqlite3.connect(db) as conn:
                step=json.loads(conn.execute(
                    "SELECT payload FROM trace_steps WHERE task_id='ops-root'"
                ).fetchone()[0])
            self.assertEqual(step["next"]["line"],2)
            self.assertEqual(step["next"]["navigation_basis"],"next_source_call_candidate")
            self.assertTrue(step["navigation_review_attempted"])

    def test_nemotron_navigation_review_runs_after_unvisited_optimization_barriers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            called=[]
            async def model(stage,payload,prompt):
                if prompt == "planner": return {"focus":"trace optimization branch"}
                if prompt == "agent_trace_worker":
                    return {"claims":[{"line":2,"quote":"def call(self, x):","claim":"method"}],
                            "next":{"file":"keras/layers/dense.py","line":6,"quote":"return x"},
                            "terminal":"unresolved"}
                if prompt == "agent_trace_navigation_review":
                    called.append(payload)
                    return {"next":{"file":"keras/layers/dense.py","line":6,"quote":"return x"},
                            "terminal":"unresolved"}
                if prompt == "agent_trace_validator": return {"approved_indices":[0]}
                raise AssertionError(prompt)
            result=asyncio.run(run(db,sources,1,1,model))
            self.assertEqual(result[0]["state"],"pending")
            self.assertEqual(called,[])
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute(
                    "SELECT line FROM trace_tasks WHERE id='dense-root'"
                ).fetchone()[0],3)
                step=json.loads(conn.execute(
                    "SELECT payload FROM trace_steps WHERE task_id='dense-root'"
                ).fetchone()[0])
            self.assertFalse(step["navigation_review_attempted"])

    def test_expired_lease_is_recovered_and_source_barrier_schedules_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources,db,dense=self.fixture(root)
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE trace_tasks SET state='running' WHERE id='dense-root'")
                conn.execute("INSERT INTO trace_leases VALUES(?,?,1)",("dense-root","expired"))
            async def model(stage,payload,prompt):
                if prompt=="planner": return {"focus":"trace optimization branch"}
                if prompt=="agent_trace_worker":
                    return {"claims":[{"line":1,"quote":"class Dense(Layer):","claim":"layer declaration"}],
                            "next":{"file":"keras/layers/dense.py","line":4,"quote":"return x + 1"},
                            "terminal":"unresolved"}
                return {"approved_indices":[0]}
            result=asyncio.run(run(db,sources,1,1,model))
            self.assertEqual(result[0]["state"],"pending")
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute("SELECT line FROM trace_tasks WHERE id='dense-root'").fetchone()[0],3)
                self.assertEqual(conn.execute("SELECT count(*) FROM trace_leases").fetchone()[0],0)


if __name__ == "__main__":
    unittest.main()
