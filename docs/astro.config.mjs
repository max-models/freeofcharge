// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/freeofcharge/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'freeofcharge',
	title: 'freeofcharge',
	description: 'Neutral beams in tokamak plasmas: deposition, shine-through, slowing down and current drive.',
	pythonPackage: 'freeofcharge',
	// the method cards of the numerical methods: a Numerics section on their API pages and /numerics/
	// the method cards: a Numerics section with the overview (numerics/index.mdx) and one page per
	// module, in the order of the model, and a Numerics section on each method's API page
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'freeofcharge.equilibrium': 'Equilibrium',
			'freeofcharge.gvec_equilibrium': 'GVEC equilibria',
			'freeofcharge.gridded_equilibrium': 'Gridded equilibria',
			'freeofcharge.vmec_equilibrium': 'VMEC equilibria',
			'freeofcharge.profiles': 'Plasma profiles',
			'freeofcharge.beam': 'Beam markers',
			'freeofcharge.stopping': 'Beam stopping',
			'freeofcharge.stopping_tables': 'Stopping tables',
			'freeofcharge.deposition': 'Deposition',
			'freeofcharge.slowing_down': 'Slowing down',
			'freeofcharge.time_dependent': 'Beam modulation',
			'freeofcharge.distribution': 'Fast-ion distribution',
			'freeofcharge.torque': 'Beam torque',
			'freeofcharge.neutrons': 'Neutron rates',
			'freeofcharge.charge_exchange': 'Charge-exchange losses',
			'freeofcharge.trapping': 'Orbit-averaged pitch',
			'freeofcharge.orbit_width': 'Orbit width',
			'freeofcharge.orbits': 'Orbits',
			'freeofcharge.shine_through': 'Shine-through loads',
			'freeofcharge.chain': 'Injector chain',
			'freeofcharge.transport_geometry': 'Transport geometry',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'Numerics', link: '/numerics/' },
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
