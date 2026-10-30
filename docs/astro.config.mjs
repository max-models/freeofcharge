// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/freeofcharge/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'freeofcharge',
	title: 'freeofcharge',
	description: 'Neutral beams in tokamak plasmas: deposition, shine-through, slowing down and current drive.',
	pythonPackage: 'freeofcharge',
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/freeofcharge/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Examples',
			items: [
				{ autogenerate: { directory: 'examples' } },
				{ label: 'Scripts', items: [{ autogenerate: { directory: 'example-scripts' } }] },
			],
		},
		{
			label: 'Benchmarks',
			items: [{ label: 'ASCOT5', slug: 'benchmarks/ascot5' }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
				{ label: 'Chain test', slug: 'development/chain-test' },
			],
		},
	],
});
