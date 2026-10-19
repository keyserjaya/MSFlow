<script lang="ts">
	import { onMount, onDestroy } from "svelte";
	import * as THREE from "three";
	import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
	import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
	import type {
		AxisName,
		CanvasEventPayload,
		CanvasValue,
		ConstraintCanvasEvents,
		ConstraintCanvasProps,
		Playback
	} from "./types";
	import { NUM_JOINTS, solvePose, type SolveTarget } from "./ik";
	import { Gradio } from "@gradio/utils";
	import { Block } from "@gradio/atoms";
	import { StatusTracker } from "@gradio/statustracker";

	const props = $props();
	const gradio = new Gradio<ConstraintCanvasEvents, ConstraintCanvasProps>(props);

	const container = true;
	const BUILD = "b7";
	const LOG = (...args: unknown[]): void =>
		console.log(`[canvas ${BUILD}]`, ...args);

	// dispatching an event does NOT feed the payload to the python handler —
	// the event's input is the component's *value*. So the payload must become
	// the value first (the $effect guard below keeps it from touching the scene).
	function dispatchEvent(payload: CanvasEventPayload): void {
		LOG("dispatch", payload);
		gradio.update({ value: payload });
		gradio.dispatch("change", payload);
	}
	const DEFAULT: CanvasValue = {
		frame: 0,
		underlay: { points: [], edges: [] },
		markers: [],
		lines: [],
		selected_joint: null,
		label: ""
	};

	let viewport: HTMLDivElement | undefined = $state();
	let ready = $state(false);
	let hudLabel = $state("");
	let isEmpty = $state(true);
	let currentFrame = $state(0);
	let syncedOnce = false;
	let lastSyncSignature = "";

	let renderer: THREE.WebGLRenderer;
	let scene: THREE.Scene;
	let camera: THREE.PerspectiveCamera;
	let orbit: OrbitControls;
	let transform: TransformControls;

	// armature base (solved pose — follows the constraints)
	const baseMeshes = new Map<number, THREE.Mesh>();
	let baseLines: THREE.LineSegments;
	let baseEdges: [number, number][] = [];
	let basePositions: (THREE.Vector3 | undefined)[] = [];

	// constraint handles (one per saved constraint, any frame)
	const handleMeshes = new Map<string, THREE.Mesh>();
	let markerInfo = new Map<
		string,
		{
			frame: number;
			joint: number;
			axes: AxisName[];
			color: string;
			controlled: boolean;
			waypoint: boolean;
			pos: THREE.Vector3;
		}
	>();

	let trajectoryGroup: THREE.Group;
	let resizeObserver: ResizeObserver;
	let frameHandle = 0;

	const raycaster = new THREE.Raycaster();
	const pointer = new THREE.Vector2();
	const dragOrigin = new THREE.Vector3();
	let pointerDown = { x: 0, y: 0 };
	let gizmoDragged = false;
	let pressedOnGizmo = false;

	// live "estimated result" solve
	let solveSpec: SolveTarget[] = [];
	let solvedPose: THREE.Vector3[] = [];
	let solveDirty = true;
	let dragTarget: { joint: number; pos: THREE.Vector3; axes: AxisName[] } | null = null;

	// client-authoritative selection: (joint, frame); no server round-trip
	let selected: { joint: number; frame: number } | null = $state(null);

	// generated-result playback (animated skeleton in the same viewport)
	let playbackFrames: number[][][] = [];
	let resultEdges: [number, number][] = [];
	let lastResultIndex = -1;
	let playbackKey = "";
	let playFrame = $state(0);
	let playing = $state(false);
	let playLen = $state(0);
	let playFps = 20;
	let playLabel = $state("");
	const resultMeshes = new Map<number, THREE.Mesh>();
	let resultLines: THREE.LineSegments | null = null;

	const GRAY = 0x8b95a8;
	const SELECTED_EMISSIVE = 0x6b5a10;
	const HANDLE_COLOR_FALLBACK = "#CC79A7";
	const GHOST_OPACITY = 0.3;
	const PLAYBACK_OPACITY = 0.45;

	function handleId(frame: number, joint: number): string {
		return `f${frame}-j${joint}`;
	}

	function normalize(raw: unknown): CanvasValue {
		const value = (raw ?? {}) as Partial<CanvasValue>;
		return {
			frame: value.frame ?? undefined,
			underlay: value.underlay ?? DEFAULT.underlay,
			markers: value.markers ?? [],
			lines: value.lines ?? [],
			selected_joint: value.selected_joint ?? null,
			label: value.label ?? "",
			playback: value.playback ?? null
		};
	}

	function axesFromTransform(axis: string | null, fallback?: AxisName[]): AxisName[] {
		if (axis) {
			const names: AxisName[] = [];
			for (const char of axis.toLowerCase()) {
				if (char === "x" || char === "y" || char === "z") names.push(char);
			}
			if (names.length) return names;
		}
		return fallback && fallback.length ? fallback.slice() : ["x", "y", "z"];
	}

	function initScene(): void {
		const width = viewport!.clientWidth || 640;
		const height = viewport!.clientHeight || 460;

		renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
		renderer.setPixelRatio(window.devicePixelRatio);
		renderer.setSize(width, height, false);
		renderer.domElement.style.width = "100%";
		renderer.domElement.style.height = "100%";
		renderer.domElement.style.display = "block";
		renderer.domElement.style.touchAction = "none";
		renderer.domElement.style.outline = "none";
		viewport!.appendChild(renderer.domElement);

		scene = new THREE.Scene();
		scene.background = new THREE.Color(0xf4f6fa);

		camera = new THREE.PerspectiveCamera(50, width / height, 0.05, 80);
		camera.position.set(2.9, 2.2, 3.6);

		orbit = new OrbitControls(camera, renderer.domElement);
		orbit.target.set(0, 0.9, 0);
		orbit.enableDamping = true;
		orbit.dampingFactor = 0.12;
		orbit.update();

		const grid = new THREE.GridHelper(8, 32, 0x9aa7b8, 0xd7dde6);
		const gridMat = grid.material as THREE.Material;
		gridMat.transparent = true;
		gridMat.opacity = 0.7;
		scene.add(grid);
		scene.add(new THREE.AxesHelper(1.3));

		scene.add(new THREE.HemisphereLight(0xffffff, 0x3a4656, 1.15));
		const sun = new THREE.DirectionalLight(0xffffff, 1.5);
		sun.position.set(4, 7, 3);
		scene.add(sun);

		baseLines = new THREE.LineSegments(
			new THREE.BufferGeometry(),
			new THREE.LineBasicMaterial({
				color: 0x8d97ad,
				transparent: true,
				opacity: 0.6
			})
		);
		scene.add(baseLines);

		trajectoryGroup = new THREE.Group();
		scene.add(trajectoryGroup);

		resultLines = new THREE.LineSegments(
			new THREE.BufferGeometry(),
			new THREE.LineBasicMaterial({
				color: 0xe67e22,
				transparent: true,
				opacity: 0.95
			})
		);
		resultLines.visible = false;
		scene.add(resultLines);
		for (let joint = 0; joint < NUM_JOINTS; joint++) {
			const mesh = new THREE.Mesh(
				new THREE.SphereGeometry(0.05, 16, 12),
				new THREE.MeshStandardMaterial({ color: 0xe67e22, roughness: 0.45 })
			);
			mesh.visible = false;
			scene.add(mesh);
			resultMeshes.set(joint, mesh);
		}

		transform = new TransformControls(camera, renderer.domElement);
		const helperHolder = transform as unknown as {
			getHelper?: () => THREE.Object3D;
		} & THREE.Object3D;
		scene.add(
			typeof helperHolder.getHelper === "function"
				? helperHolder.getHelper()
				: (transform as unknown as THREE.Object3D)
		);
		transform.setMode("translate");
		transform.setSize(0.85);
		transform.addEventListener("dragging-changed", (event: unknown) => {
			const e = event as { value: boolean };
			orbit.enabled = !e.value;
			gizmoDragged = e.value;
			if (e.value && transform.object) {
				dragOrigin.copy((transform.object as THREE.Mesh).position);
				dragTarget = null;
			}
		});
		transform.addEventListener("objectChange", () => {
			if (!transform.dragging || !transform.object) return;
			const mesh = transform.object as THREE.Mesh;
			dragTarget = {
				joint: Number(mesh.userData.joint),
				pos: mesh.position.clone(),
				axes: axesFromTransform(transform.axis, mesh.userData.axes as AxisName[] | undefined)
			};
			solveDirty = true;
		});
		transform.addEventListener("mouseUp", () => {
			const mesh = transform.object as THREE.Mesh | undefined;
			if (!mesh) return;
			const joint = Number(mesh.userData.joint);
			const end = mesh.position;
			const touched: AxisName[] = [];
			if (Math.abs(end.x - dragOrigin.x) > 1e-5) touched.push("x");
			if (Math.abs(end.y - dragOrigin.y) > 1e-5) touched.push("y");
			if (Math.abs(end.z - dragOrigin.z) > 1e-5) touched.push("z");
			if (!touched.length) return;
			if (dragTarget) dragTarget.axes = touched;
			// a raw-joint drag creates a constraint: materialize its handle locally so
			// it stays visible until the next server-driven value sync
			const wasHandle = mesh.userData.handleId !== undefined;
			if (!wasHandle) {
				const id = handleId(currentFrame, joint);
				let handle = handleMeshes.get(id);
				if (!handle) {
					handle = newMeshHandle(id, joint, end, touched, HANDLE_COLOR_FALLBACK);
					scene.add(handle);
					handleMeshes.set(id, handle);
					markerInfo.set(id, {
						frame: currentFrame,
						joint,
						axes: touched,
						color: HANDLE_COLOR_FALLBACK,
						controlled: true,
						pos: end.clone()
					});
				}
				applyHandleStyles();
				if (transform.object === mesh) {
					transform.attach(handleMeshes.get(id)!);
				}
			}
			const finalAxes = touched;
			const payload: CanvasEventPayload = {
				action: "move",
				joint,
				x: end.x,
				y: end.y,
				z: end.z,
				axes: finalAxes,
				frame: currentFrame
			};
			LOG("commit move", { ...payload, dragOrigin: dragOrigin.toArray() });
			dispatchEvent(payload);
		});

		renderer.domElement.addEventListener("pointerdown", onPointerDown);
		renderer.domElement.addEventListener("pointerup", onPointerUp);
		window.addEventListener("keydown", onKeyDown);

		resizeObserver = new ResizeObserver(() => {
			const w = viewport?.clientWidth ?? 0;
			const h = viewport?.clientHeight ?? 0;
			if (!w || !h) return;
			camera.aspect = w / h;
			camera.updateProjectionMatrix();
			renderer.setSize(w, h, false);
		});
		resizeObserver.observe(viewport!);

		const loop = (now: number) => {
			frameHandle = requestAnimationFrame(loop);
			const dt = Math.min(0.1, (now - lastTick) / 1000);
			lastTick = now;
			if (playing && playLen > 1) {
				playFrame = (playFrame + dt * playFps) % playLen;
			}
			refreshSolve();
			updateResultPose();
			orbit.update();
			renderer.render(scene, camera);
		};
		let lastTick = performance.now();
		frameHandle = requestAnimationFrame(loop);
		ready = true;
	}

	function newMeshHandle(
		id: string,
		joint: number,
		pos: THREE.Vector3,
		axes: AxisName[],
		color: string,
		waypoint = false
	): THREE.Mesh {
		const mesh = new THREE.Mesh(
			new THREE.SphereGeometry(waypoint ? 0.075 : 0.055, 20, 16),
			new THREE.MeshStandardMaterial({
				color,
				roughness: 0.4,
				metalness: 0.05,
				transparent: true
			})
		);
		mesh.userData.handleId = id;
		mesh.userData.joint = joint;
		mesh.userData.axes = axes;
		mesh.position.copy(pos);
		if (waypoint) {
			// path pins sit inside the body: draw through meshes so they stay
			// visible/grabbable above the armature
			(mesh.material as THREE.MeshStandardMaterial).depthTest = false;
			mesh.renderOrder = 10;
		}
		return mesh;
	}

	function onPointerDown(event: PointerEvent): void {
		pointerDown = { x: event.clientX, y: event.clientY };
		// captured at press time: three.js nulls its axis/dragging state before
		// our pointerup listener runs, so those can't be trusted at release
		pressedOnGizmo = transform.axis !== null || transform.dragging;
	}

	function onPointerUp(event: PointerEvent): void {
		const wasGizmo = pressedOnGizmo || gizmoDragged;
		pressedOnGizmo = false;
		gizmoDragged = false;
		if (wasGizmo) {
			LOG("pointerup: gizmo interaction, skip raycast");
			return;
		}
		const moved = Math.hypot(
			event.clientX - pointerDown.x,
			event.clientY - pointerDown.y
		);
		if (moved > 6) {
			LOG("pointerup: camera drag", { moved });
			return;
		}

		const rect = renderer.domElement.getBoundingClientRect();
		pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
		pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
		raycaster.setFromCamera(pointer, camera);

		const handleHits = raycaster.intersectObjects([...handleMeshes.values()], false);
		if (handleHits.length) {
			selectHandle(String(handleHits[0].object.userData.handleId));
			return;
		}
		const baseHits = raycaster.intersectObjects([...baseMeshes.values()], false);
		if (baseHits.length) {
			selectBaseJoint(Number(baseHits[0].object.userData.joint));
			return;
		}
		deselect();
	}

	function onKeyDown(event: KeyboardEvent): void {
		if (event.key === "Escape" && selected !== null) {
			deselect();
		}
	}

	function selectHandle(id: string): void {
		const info = markerInfo.get(id);
		if (!info) return;
		LOG("click handle", { id, frame: info.frame, joint: info.joint, selected });
		if (info.frame === currentFrame) {
			// selecting (re-)attaches the drag gizmo; deselecting is a click on
			// empty space or Escape — re-clicking must never yank the gizmo
			// away from someone trying to grab the marker
			setSelection({ joint: info.joint, frame: currentFrame });
			dispatchEvent({
				action: "select",
				joint: info.joint,
				frame: currentFrame
			});
		} else {
			// ghost of another frame: jump the editing frame there (server switches)
			setSelection(null);
			dispatchEvent({
				action: "select",
				joint: info.joint,
				frame: info.frame
			});
		}
	}

	function selectBaseJoint(joint: number): void {
		LOG("click base joint", { joint, selected });
		// selecting always (re-)attaches; empty-space click / Escape deselects
		setSelection({ joint, frame: currentFrame });
		dispatchEvent({
			action: "select",
			joint,
			frame: currentFrame
		});
	}

	function deselect(): void {
		const was = selected;
		LOG("deselect", { was });
		setSelection(null);
		if (was === null) return; // already clear — no round-trip needed
		dispatchEvent({
			action: "select",
			joint: -1,
			frame: currentFrame
		});
	}

	function setSelection(sel: { joint: number; frame: number } | null): void {
		selected = sel;
		if (!sel) {
			transform.detach();
		} else if (sel.frame === currentFrame) {
			const handle = handleMeshes.get(handleId(sel.frame, sel.joint));
			if (handle) {
				transform.attach(handle);
			} else if (baseMeshes.has(sel.joint)) {
				transform.attach(baseMeshes.get(sel.joint)!);
			} else {
				transform.detach();
			}
		} else {
			transform.detach();
		}
		updateSelectionVisuals();
	}

	function selectionResolvable(sel: { joint: number; frame: number } | null): boolean {
		if (!sel) return true;
		if (handleMeshes.has(handleId(sel.frame, sel.joint))) return true;
		return sel.frame === currentFrame && baseMeshes.has(sel.joint);
	}

	function setHighlighted(mesh: THREE.Mesh, on: boolean): void {
		const mat = mesh.material as THREE.MeshStandardMaterial;
		mat.emissive.setHex(on ? SELECTED_EMISSIVE : 0x000000);
		mesh.scale.setScalar(on ? 1.35 : 1);
	}

	function updateSelectionVisuals(): void {
		const selId = selected ? handleId(selected.frame, selected.joint) : null;
		const hasHandle = selId !== null && handleMeshes.has(selId);
		const baseJoint =
			selected && selected.frame === currentFrame && !hasHandle
				? selected.joint
				: Number.NaN;
		for (const [joint, mesh] of baseMeshes) setHighlighted(mesh, joint === baseJoint);
		for (const [id, mesh] of handleMeshes) setHighlighted(mesh, id === selId);
	}

	function applyHandleStyles(): void {
		const playbackActive = playLen > 0;
		for (const [id, mesh] of handleMeshes) {
			const info = markerInfo.get(id);
			if (!info) continue;
			const mat = mesh.material as THREE.MeshStandardMaterial;
			const opacity = playbackActive
				? PLAYBACK_OPACITY
				: info.frame === currentFrame
					? 1.0
					: GHOST_OPACITY;
			mat.opacity = opacity;
			mat.depthWrite = opacity >= 0.95;
		}
	}

	function refreshSolve(): void {
		if (isEmpty) {
			return;
		}
		if (solveDirty) {
			const targets: SolveTarget[] = solveSpec.slice();
			if (dragTarget) {
				const index = targets.findIndex((t) => t.joint === dragTarget!.joint);
				if (index >= 0) targets[index] = { ...dragTarget! };
				else targets.push({ ...dragTarget! });
			}
			const dense: THREE.Vector3[] = [];
			for (let joint = 0; joint < NUM_JOINTS; joint++) {
				dense.push(basePositions[joint] ?? new THREE.Vector3());
			}
			solvedPose = solvePose(dense, targets, 12);
			solveDirty = false;
			for (const [joint, mesh] of baseMeshes) {
				if (transform.dragging && transform.object === mesh) continue;
				const p = solvedPose[joint];
				if (p) mesh.position.copy(p);
			}
			if (baseLines) {
				const verts: number[] = [];
				for (const [a, b] of baseEdges) {
					const pa = solvedPose[a];
					const pb = solvedPose[b];
					if (!pa || !pb) continue;
					verts.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
				}
				baseLines.geometry.dispose();
				const geo = new THREE.BufferGeometry();
				geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
				baseLines.geometry = geo;
			}
		}
	}

	function updateResultPose(): void {
		if (!playbackFrames.length || playLen === 0) return;
		const index = Math.min(
			playLen - 1,
			Math.max(0, Math.round(playFrame))
		);
		if (index === lastResultIndex) return;
		lastResultIndex = index;
		const frame = playbackFrames[index];
		if (!frame) return;
		for (const [joint, mesh] of resultMeshes) {
			const p = frame[joint];
			if (p) mesh.position.set(p[0], p[1] ?? 0, p[2]);
		}
		if (resultLines) {
			const verts: number[] = [];
			for (const [a, b] of resultEdges) {
				const pa = frame[a];
				const pb = frame[b];
				if (!pa || !pb) continue;
				verts.push(pa[0], pa[1] ?? 0, pa[2], pb[0], pb[1] ?? 0, pb[2]);
			}
			resultLines.geometry.dispose();
			const geo = new THREE.BufferGeometry();
			geo.setAttribute(
				"position",
				new THREE.Float32BufferAttribute(verts, 3)
			);
			resultLines.geometry = geo;
		}
	}

	function syncPlayback(value: CanvasValue): void {
		const playback: Playback | null | undefined = value.playback;
		const frames = playback?.frames ?? [];
		if (frames.length) {
			const key = `${playback?.label ?? ""}:${frames.length}`;
			playbackFrames = frames;
			playLen = frames.length;
			playFps = playback?.fps ?? 20;
			playLabel = playback?.label ?? "";
			resultEdges = value.underlay.edges ?? [];
			for (const mesh of resultMeshes.values()) mesh.visible = true;
			if (resultLines) resultLines.visible = true;
			if (key !== playbackKey) {
				playbackKey = key;
				playFrame = 0;
				lastResultIndex = -1;
				playing = true;
			}
		} else {
			playbackFrames = [];
			playLen = 0;
			playing = false;
			playLabel = "";
			playbackKey = "";
			for (const mesh of resultMeshes.values()) mesh.visible = false;
			if (resultLines) resultLines.visible = false;
			lastResultIndex = -1;
		}
		applyHandleStyles();
		updateResultPose();
	}

	function snapshotFrame(): void {
		let pose: number[][];
		if (playbackFrames.length && playLen > 0) {
			// pin the pose the user is LOOKING at: the paused animation frame,
			// not the solved editing armature
			const index = Math.min(
				playLen - 1,
				Math.max(0, Math.round(playFrame))
			);
			const frame = playbackFrames[index] ?? [];
			if (frame.length < NUM_JOINTS) return;
			pose = frame.slice(0, NUM_JOINTS).map((p) => [
				Math.round((p[0] ?? 0) * 1e4) / 1e4,
				Math.round((p[1] ?? 0) * 1e4) / 1e4,
				Math.round((p[2] ?? 0) * 1e4) / 1e4
			]);
		} else {
			if (!solvedPose.length) return;
			solveDirty = true;
			refreshSolve();
			pose = solvedPose.slice(0, NUM_JOINTS).map((p) => [
				Math.round(p.x * 1e4) / 1e4,
				Math.round(p.y * 1e4) / 1e4,
				Math.round(p.z * 1e4) / 1e4
			]);
		}
		dispatchEvent({ action: "snapshot", frame: currentFrame, pose });
		LOG("snapshot", {
			source: playbackFrames.length && playLen > 0 ? "playback" : "solved",
			playFrame: Math.round(playFrame),
			currentFrame,
			rows: pose.length,
			pose0: pose[0]
		});
	}

	function syncScene(value: CanvasValue): void {
		hudLabel = value.label ?? "";
		const prevFrame = currentFrame;
		const frameSwitch = !syncedOnce || prevFrame !== (value.frame ?? 0);
		syncedOnce = true;
		currentFrame = value.frame ?? 0;

		// base pose at the current frame (the armature's FK start), indexed by joint id
		basePositions = new Array(NUM_JOINTS).fill(undefined);
		for (const p of value.underlay.points) {
			basePositions[Number(p.joint)] = new THREE.Vector3(p.x, p.y ?? 0, p.z);
		}
		baseEdges = value.underlay.edges;
		isEmpty = basePositions.every((v) => !v);
		if (isEmpty && baseLines) {
			baseLines.geometry.dispose();
			baseLines.geometry = new THREE.BufferGeometry();
		}

		// handles: every saved constraint (all frames), later entries win on id collision
		const markersById = new Map<string, CanvasValue["markers"][number]>();
		for (const m of value.markers) {
			markersById.set(m.id, m);
		}
		markerInfo = new Map();
		for (const [id, m] of markersById) {
			markerInfo.set(id, {
				frame: m.frame,
				joint: m.joint,
				axes: m.axes && m.axes.length ? m.axes : ["x", "y", "z"],
				color: m.color ?? HANDLE_COLOR_FALLBACK,
				controlled: m.controlled !== false,
				waypoint: m.waypoint === true,
				pos: new THREE.Vector3(m.x, m.y ?? 0, m.z)
			});
		}

		for (const [id, m] of markersById) {
			const info = markerInfo.get(id);
			if (!info) continue;
			let mesh = handleMeshes.get(id);
			if (!mesh) {
				mesh = newMeshHandle(id, m.joint, info.pos, info.axes, info.color, info.waypoint);
				scene.add(mesh);
				handleMeshes.set(id, mesh);
			}
			if (!(transform.dragging && transform.object === mesh)) {
				mesh.position.copy(info.pos);
			}
			(mesh.material as THREE.MeshStandardMaterial).color.set(info.color);
		}
		for (const [id, mesh] of [...handleMeshes]) {
			if (!markerInfo.has(id)) {
				if (transform.object === mesh) transform.detach();
				scene.remove(mesh);
				mesh.geometry.dispose();
				(mesh.material as THREE.Material).dispose();
				handleMeshes.delete(id);
			}
		}

		// base joint meshes (positioned by the solver)
		for (let joint = 0; joint < basePositions.length; joint++) {
			const base = basePositions[joint];
			if (!base) continue;
			let mesh = baseMeshes.get(joint);
			if (!mesh) {
				mesh = new THREE.Mesh(
					new THREE.SphereGeometry(0.05, 20, 16),
					new THREE.MeshStandardMaterial({ color: GRAY, roughness: 0.45, metalness: 0.05 })
				);
				mesh.userData.joint = joint;
				mesh.position.copy(base);
				scene.add(mesh);
				baseMeshes.set(joint, mesh);
			}
		}
		for (const [joint, mesh] of [...baseMeshes]) {
			if (!basePositions[joint]) {
				if (transform.object === mesh) transform.detach();
				scene.remove(mesh);
				mesh.geometry.dispose();
				(mesh.material as THREE.Material).dispose();
				baseMeshes.delete(joint);
			}
		}

		// solve targets: every constraint at the current frame
		solveSpec = [];
		for (const info of markerInfo.values()) {
			if (info.frame !== currentFrame) continue;
			solveSpec.push({
				joint: info.joint,
				pos: info.pos.clone(),
				axes: info.axes
			});
		}
		dragTarget = null; // server state is fresh; a new drag rebuilds it
		solveDirty = true;

		// trajectory lines (unchanged semantics: joints of the selected control)
		while (trajectoryGroup.children.length) {
			const child = trajectoryGroup.children.pop()!;
			(child as THREE.Line).geometry?.dispose();
			(child as THREE.Line).material instanceof THREE.Material &&
				(child as THREE.Line).material.dispose();
		}
		for (const line of value.lines) {
			if (!line.points || line.points.length < 2) continue;
			const pts = line.points.map(
				(p) => new THREE.Vector3(p[0], p[1] ?? 0, p[2])
			);
			const geo = new THREE.BufferGeometry().setFromPoints(pts);
			const mat = new THREE.LineDashedMaterial({
				color: line.color ?? HANDLE_COLOR_FALLBACK,
				dashSize: 0.07,
				gapSize: 0.055,
				transparent: true,
				opacity: 0.9
			});
			const dashed = new THREE.Line(geo, mat);
			dashed.computeLineDistances();
			trajectoryGroup.add(dashed);
		}

		syncPlayback(value);

		// selection sync (client-authoritative): adopt the server's selection
		// when nothing is selected locally (button-driven re-issues like the
		// waypoint/keyframe buttons should arm the gizmo immediately; ghost
		// jumps arrive the same way) or when the local selection can no longer
		// resolve. A same-frame response NEVER steals a live selection the
		// user already has — only null selections get armed.
		const serverSel = value.selected_joint ?? null;
		let decision = "keep-client";
		if (selected !== null && !selectionResolvable(selected)) {
			if (serverSel !== null) {
				decision = "adopt-server(unresolvable)";
				setSelection({ joint: serverSel, frame: currentFrame });
			} else {
				decision = "clear(unresolvable)";
				setSelection(null);
			}
		} else if (selected === null && serverSel !== null) {
			decision = frameSwitch
				? "adopt-server(frame-switch)"
				: "adopt-server(button)";
			setSelection({ joint: serverSel, frame: currentFrame });
		} else {
			if (selected !== null) {
				// re-attach in case meshes were rebuilt
				setSelection(selected);
			}
			updateSelectionVisuals();
		}
		LOG("sync", {
			frame: currentFrame,
			prevFrame,
			frameSwitch,
			markers: value.markers.length,
			serverSel,
			selected,
			decision,
			playback: value.playback?.frames?.length ?? 0
		});
	}

	$effect(() => {
		const raw = gradio.props.value;
		if (
			raw &&
			typeof raw === "object" &&
			!Array.isArray(raw) &&
			!("underlay" in (raw as object))
		) {
			// a dispatched action payload momentarily became the component value —
			// keep the live scene and selection instead of tearing everything down
			LOG("value is an event payload — ignored");
			return;
		}
		const value = normalize(raw);
		let signature = "";
		try {
			signature = JSON.stringify(value);
		} catch {
			signature = String(Math.random());
		}
		if (signature === lastSyncSignature) {
			// gradio re-delivers the last value after each event settles — a
			// rebuild from an unchanged value is what "reset" dragged positions
			LOG("value unchanged — rebuild skipped");
			return;
		}
		lastSyncSignature = signature;
		if (ready) {
			syncScene(value);
		} else {
			hudLabel = value.label ?? "";
		}
	});

	onMount(() => {
		initScene();
	});

	onDestroy(() => {
		cancelAnimationFrame(frameHandle);
		resizeObserver?.disconnect();
		renderer?.domElement.removeEventListener("pointerdown", onPointerDown);
		renderer?.domElement.removeEventListener("pointerup", onPointerUp);
		window.removeEventListener("keydown", onKeyDown);
		for (const mesh of [...baseMeshes.values(), ...handleMeshes.values(), ...resultMeshes.values()]) {
			mesh.geometry?.dispose();
			(mesh.material as THREE.Material)?.dispose();
		}
		renderer?.dispose();
	});
</script>

<Block
	visible={gradio.shared.visible}
	elem_id={gradio.shared.elem_id}
	elem_classes={gradio.shared.elem_classes}
	{container}
	scale={gradio.shared.scale}
	min_width={gradio.shared.min_width}
>
	{#if gradio.shared.loading_status}
		<StatusTracker
			autoscroll={gradio.shared.autoscroll}
			i18n={gradio.i18n}
			{...gradio.shared.loading_status}
			on_clear_status={() =>
				gradio.dispatch("clear_status", gradio.shared.loading_status)}
		/>
	{/if}
	<div class="scene-wrap" data-testid="constraint-canvas">
		<div class="viewport" bind:this={viewport}></div>
		{#if !isEmpty}
			<button
				class="pin"
				type="button"
				onclick={snapshotFrame}
				title="pin the whole current pose (after solve) as constraints at the current frame"
			>
				🎯 constrain this frame
			</button>
		{/if}
		<div class="hud">
			<span class="hint"
				>left-drag: orbit · right-drag: pan · scroll: zoom · click joint or
				constraint: select · drag gizmo arrows/planes: move · esc: deselect</span
			>
			{#if hudLabel}<span class="label">{hudLabel}</span>{/if}
		</div>
		{#if playLen > 1}
			<div class="playbar">
				<button class="play" onclick={() => (playing = !playing)}>
					{playing ? "⏸" : "▶"}
				</button>
				<input
					class="scrub"
					type="range"
					min="0"
					max={playLen - 1}
					step="1"
					value={Math.round(playFrame)}
					oninput={(event) => {
						playing = false;
						playFrame = Number((event.target as HTMLInputElement).value);
					}}
				/>
				<span class="frame">
					{playLabel ? `${playLabel} · ` : ""}f {Math.round(playFrame) + 1}/{playLen}
				</span>
			</div>
		{/if}
		{#if isEmpty}
			<div class="empty">no pose loaded — pick a reference motion</div>
		{/if}
	</div>
</Block>

<style>
	.scene-wrap {
		position: relative;
		border: 1px solid var(--border-color-primary, #e2e5ec);
		border-radius: var(--radius-lg, 8px);
		overflow: hidden;
		background: #f4f6fa;
	}
	.viewport {
		height: 460px;
		width: 100%;
	}
	.pin {
		position: absolute;
		top: 8px;
		left: 8px;
		padding: 5px 10px;
		border: 1px solid #c3ccd9;
		border-radius: 8px;
		background: rgba(255, 255, 255, 0.92);
		cursor: pointer;
		font-size: 12px;
		color: #444c5c;
	}
	.pin:hover {
		background: #eef2f8;
	}
	.hud {
		position: absolute;
		left: 0;
		right: 0;
		bottom: 0;
		display: flex;
		justify-content: space-between;
		gap: 12px;
		padding: 6px 10px;
		font-size: 11px;
		color: #5a6474;
		background: linear-gradient(transparent, rgba(244, 246, 250, 0.92));
		pointer-events: none;
	}
	.label {
		font-weight: 600;
		white-space: nowrap;
	}
	.empty {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		color: #8b95a8;
		font-size: 13px;
		pointer-events: none;
	}
	.playbar {
		position: absolute;
		top: 8px;
		right: 8px;
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 5px 10px;
		border-radius: 8px;
		background: rgba(255, 255, 255, 0.92);
		border: 1px solid #d7dde6;
		font-size: 11px;
		color: #444c5c;
	}
	.play {
		width: 26px;
		height: 22px;
		border: 1px solid #c3ccd9;
		border-radius: 5px;
		background: #fff;
		cursor: pointer;
		font-size: 11px;
		line-height: 1;
	}
	.play:hover {
		background: #eef2f8;
	}
	.scrub {
		width: 150px;
		accent-color: #e67e22;
	}
	.frame {
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
</style>
