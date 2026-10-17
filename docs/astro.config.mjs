// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/template-python/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'template-python',
	title: 'template-python',
	description: 'Template repository for Python projects.',
	pythonPackage: 'app',
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Tutorials', link: '/tutorials/' },
		{ label: 'Full examples', link: '/full-examples/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/app/' },
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
			label: 'Tutorials',
			items: [{ autogenerate: { directory: 'tutorials' } }],
		},
		{
			label: 'Full examples',
			items: [{ autogenerate: { directory: 'full-examples' } }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});
