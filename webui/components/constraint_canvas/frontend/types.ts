import type { LoadingStatus } from "@gradio/utils";

export type AxisName = "x" | "y" | "z";

export interface CanvasPoint {
	joint: number;
	x: number;
	y?: number;
	z: number;
}

export interface CanvasMarker {
	id: string; // stable handle id, e.g. "f60-j21"
	joint: number;
	frame: number; // the constraint's frame (may differ from the current frame)
	x: number;
	y?: number;
	z: number;
	axes?: AxisName[]; // which position axes are constrained
	color?: string;
	controlled?: boolean;
	selected?: boolean;
}

export interface CanvasLine {
	points: [number, number, number][];
	color?: string;
}

export interface Playback {
	frames: number[][][]; // [T][22][3] world joints of the generated result
	fps?: number;
	label?: string;
}

export interface CanvasValue {
	frame?: number; // current editing frame
	underlay: { points: CanvasPoint[]; edges: [number, number][] };
	markers: CanvasMarker[];
	lines: CanvasLine[];
	selected_joint: number | null;
	label: string;
	playback?: Playback | null;
}

export type CanvasEventPayload =
	| {
			action: "move";
			joint: number;
			x: number;
			y: number;
			z: number;
			axes: AxisName[];
			frame?: number; // the editing frame the drag happened at
	  }
	| { action: "select"; joint: number; frame?: number } // frame set = ghost jump; -1 deselects
	| { action: "snapshot"; frame: number; pose: number[][] }; // 22×[x,y,z] solved pose

export interface ConstraintCanvasProps {
	value: CanvasValue | null;
	show_label?: boolean;
}

export interface ConstraintCanvasEvents {
	change: CanvasEventPayload;
	clear_status: LoadingStatus;
}
