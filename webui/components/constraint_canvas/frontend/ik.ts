import * as THREE from "three";
import type { AxisName } from "./types";

// HumanML3D t2m 22-joint chain — mirror of webui/skeleton.py JOINT_CHAIN.
export const JOINT_CHAIN: [number, number][] = [
	[0, 1],
	[0, 2],
	[1, 4],
	[4, 7],
	[7, 10],
	[2, 5],
	[5, 8],
	[8, 11],
	[0, 3],
	[3, 6],
	[6, 9],
	[9, 12],
	[12, 15],
	[12, 13],
	[13, 16],
	[16, 18],
	[18, 20],
	[14, 17],
	[17, 19],
	[19, 21]
];

export const NUM_JOINTS = 22;

export interface SolveTarget {
	joint: number;
	pos: THREE.Vector3;
	axes: AxisName[]; // constrained axes; the rest stay free
}

const PARENTS: number[] = (() => {
	const parents = new Array(NUM_JOINTS).fill(-1);
	for (const [a, b] of JOINT_CHAIN) {
		parents[b] = a;
	}
	return parents;
})();

const CHILDREN: number[][] = (() => {
	const children: number[][] = Array.from({ length: NUM_JOINTS }, () => []);
	for (const [a, b] of JOINT_CHAIN) {
		children[a].push(b);
	}
	return children;
})();

function chainTo(joint: number): number[] {
	// root..joint path (root = 0); tolerates a missing parent gracefully.
	const path: number[] = [];
	let joint_i = joint;
	let guard = 0;
	while (joint_i >= 0 && joint_i < NUM_JOINTS && guard++ < NUM_JOINTS) {
		path.push(joint_i);
		joint_i = PARENTS[joint_i];
	}
	return path.reverse();
}

function rotateSubtree(
	pose: THREE.Vector3[],
	pivot: number,
	quaternion: THREE.Quaternion
): void {
	const pivotPos = pose[pivot];
	const stack = [...CHILDREN[pivot]];
	while (stack.length) {
		const joint = stack.pop()!;
		pose[joint]
			.sub(pivotPos)
			.applyQuaternion(quaternion)
			.add(pivotPos);
		for (const child of CHILDREN[joint]) stack.push(child);
	}
}

/** Quaternion rotating direction `from` onto `to`, damped by `damping`. */
function toward(
	from: THREE.Vector3,
	to: THREE.Vector3,
	damping: number
): THREE.Quaternion | null {
	if (from.lengthSq() < 1e-12 || to.lengthSq() < 1e-12) return null;
	const cross = new THREE.Vector3().crossVectors(from, to);
	const sin = cross.length();
	const cos = from.dot(to);
	if (sin < 1e-8) {
		if (cos > 0) return null; // already aligned
		// 180°: pick any orthogonal axis
		const axis = Math.abs(from.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
		cross.crossVectors(from, axis);
	}
	cross.normalize();
	const angle = damping * Math.atan2(sin, Math.max(cos, -1));
	return new THREE.Quaternion().setFromAxisAngle(cross, angle);
}

/**
 * Cheap "estimated result" solver: CCD per constrained joint over the limb
 * chain, followed by an exact projection of the constrained axes. Preserves
 * bone lengths inside every chain rotation; projection may relax them on
 * unconstrained axes (accepted — it is an estimate, not a retarget).
 *
 * `targets` are the pins at ONE frame; only constrained axes are enforced,
 * matching the spec's per-control `axes` semantics.
 */
export function solvePose(
	base: THREE.Vector3[],
	targets: SolveTarget[],
	iterations = 12
): THREE.Vector3[] {
	const pose = base.map((vector) => vector.clone());
	if (!targets.length) return pose;

	for (let iteration = 0; iteration < iterations; iteration++) {
		for (const target of targets) {
			const path = chainTo(target.joint);
			if (path.length < 2) continue;
			for (let k = path.length - 2; k >= 0; k--) {
				const pivot = path[k];
				const end = pose[target.joint].clone(); // re-read: earlier pivots moved it
				const quaternion = toward(
					end.clone().sub(pose[pivot]),
					target.pos.clone().sub(pose[pivot]),
					0.85
				);
				if (quaternion) rotateSubtree(pose, pivot, quaternion);
			}
		}
		// project pinned axes so targets are honored exactly
		for (const target of targets) {
			for (const axis of target.axes) {
				pose[target.joint][axis] = target.pos[axis];
			}
		}
	}
	return pose;
}
