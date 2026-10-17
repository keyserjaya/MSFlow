<script lang="ts">
	import type { CanvasValue } from "./types";
	import Index from "./Index.svelte";

	export let value: CanvasValue | null = null;
	export let type: "gallery" | "table" = "gallery";
	export let selected = false;

	void type;
	void selected;
</script>

<div class:table={type === "table"} class:gallery={type === "gallery"} class:selected>
	<Index {value} />
</div>

<style>
	.gallery {
		padding: var(--size-1) var(--size-2);
	}
</style>
