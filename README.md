<p align="center">
  <a href="https://github.com/lineanilt/godot/tags">
    <img src="fork_icon.svg" width="256" alt="Godot Engine logo">
  </a>
</p>

This is a **fork of Godot 3.x** that adds a bunch of new features and merges some PRs.
- Export tooltips. `## Like this.` (35716)
- NodePath export hints. (39155 - the bastards closed the PR.)
- Shadow dither. (53967 - I added onto this with some other stuff.)
- GTAO. (53886 - This has some haloing.)
- 16-bit shadowmaps. (57430)
- ViewportTexture G-Buffer access. (38926 - Slightly buggy. I'd suggest disabling and reenabling the G-Buffer at scene launch.)
	- The Normal texture's alpha has the roughness of materials.
	- If you're not using it, you can strip the Subsurface Scattering shader of its functions so its G-Buffer texture can be, say, a custom float pass.
- PCF25. (54355)
- NPOT shadowmaps (54042 - This doesn't work on shadow atlases.)
- Non-uniform light scaling. (self - shadows break when this is enabled though. I think it's the UVs, but I might be wrong.)
- Hard shadows are actually hard. (self)
- Square light mode. OmniLight shadows still render in a sphere for some reason. (self)
- Per-light shadow blur. This goes well with shader dither. (self)
- (self) Some new shader built-ins. Namely:
	- `CUSTOM_PASS_DATA` (vec4).
	- `CUSTOM_TEXTURE` to view `CUSTOM_PASS_DATA` (sampler2D). 16-bit.
		- For some reason, you need to apply an ALPHA to your shader to view it. Not sure why.
	- `textureGather`. I forgot what OpenGL version this requires.
- Some contact shadow improvements. (self)
- Depth texture can be accessed in CanvasItems. Make sure you have a proper 3D viewport, though. (self)

I probably missed some but still.

It's quite stable and Mono works.

The main point of this fork is that it's, well, Godot 3 and not Godot 4. 
People sticking to Godot 3 may have a lot of reasons for it, but for me personally, it's: 
- The switch to Vulkan.
- The direction of where Godot is headed.
- The shear ignorance of Godot Foundation.
- The fact that Godot is creating an "Asset Store" that has _monetised_ assets.
- Stability.

And I made this fork because I wanted new features without switching major engine versions.
I also wanted to stay on OpenGL, and I wanted the lightness of Godot 3.
Read Flaws for more details.

#### License
MPL2.0. Godot and its PRs were MIT.
Games you export with Godot are not automatically MPL or MIT.

### Building
You can just build this like you would a vanilla Godot 3.x version.
Though I just use this:
```bash
# Editor
scons -j4 platform=x11 target=release_debug use_llvm=yes linker=lld use_ccache=yes module_mono_enabled=yes mono_glue=yes mono_static=yes copy_mono_root=yes
# Exports
scons -j4 platform=x11 tools=no target=release_debug use_llvm=yes linker=lld use_ccache=yes module_mono_enabled=yes mono_glue=yes mono_static=yes copy_mono_root=yes
scons -j4 platform=x11 tools=no target=release use_llvm=yes linker=lld use_ccache=yes module_mono_enabled=yes mono_glue=yes mono_static=yes copy_mono_root=yes
# Windows
scons platform=windows tools=no target=release -j4 use_ccache=yes module_mono_enabled=yes mono_glue=yes bits=64 mono_prefix="/path/to/windows_mono" mono_static=yes copy_mono_root=yes

# Stripping
strip bin/godot.x11.opt.64.llvm.mono bin/godot.x11.opt.debug.64.llvm.mono bin/godot.x11.opt.tools.64.llvm.mono bin/godot.windows.opt.64.mono.exe
```
(Linux)
You'll have to download BCLs though.
You can get it here: `https://github.com/godotengine/godot-mono-builds/releases`

### Contributing
You don't _need_ to contribute, but it would be appreciated if you did.
This fork is mostly just for stylised games.
If your PR looks sick and works well, I might merge it.

###### *Side note, don't bother creating a PR for the official repo's 3.x branch. It is effectively dead and the only commits you'll see there are from @lawnjelly (<3) doing minor bugfixes. Your PRs won't get noticed and they literally close ones that \*they\* ignored because the OP was 'inactive'. Literally just explore the 3.x milestone's PR list and you'll see.*

### Usage
Only the editors are up for now.

### Flaws
- GLES2 doesn't have the new features. Well, most of them.
- This would not work on most mobile drivers.
- On 16-texunit GPUs, you only have 3 samplers available for `spatial` shaders. I'd recommend texture packing or `sampler2DArray`s. I mean, that's what you should be doing in vanilla Godot 3 anyway, not just because of the limitations, but for performance.
- Shaders using `textureGather` would fail on older GPU drivers that do not support OpenGL 4 extensions.
- It's a bit janky since it was mostly made for _me_. I decided to publish it because, well, not a lot of Godot 3 forks.
- Because of the previous point, some of my additions are AI generated. No, it's not gonna explode your PC, it works well.

Unless you're targeting _really_ old hardware that does not support GL3.3 or GL4.x (which is like... older than 2009), these aren't exactly major issues.  
And as for mobile, well, it compiles, but it wouldn't _compile shaders_ due to MRTs. 

## Original README
---
# Godot Engine

## 2D and 3D cross-platform game engine

**[Godot Engine](https://godotengine.org) is a feature-packed, cross-platform
game engine to create 2D and 3D games from a unified interface.** It provides a
comprehensive set of [common tools](https://godotengine.org/features), so that users can focus on making games
without having to reinvent the wheel. Games can be exported with one click to a
number of platforms, including the major desktop platforms (Linux, macOS,
Windows), mobile platforms (Android, iOS), as well as Web-based platforms
(HTML5) and
[consoles](https://docs.godotengine.org/en/latest/tutorials/platform/consoles.html).

## Free, open source and community-driven

Godot is completely free and open source under the very permissive [MIT license](https://godotengine.org/license).
No strings attached, no royalties, nothing. The users' games are theirs, down
to the last line of engine code. Godot's development is fully independent and
community-driven, empowering users to help shape their engine to match their
expectations. It is supported by the [Software Freedom Conservancy](https://sfconservancy.org/)
not-for-profit.

Before being open sourced in [February 2014](https://github.com/godotengine/godot/commit/0b806ee0fc9097fa7bda7ac0109191c9c5e0a1ac),
Godot had been developed by [Juan Linietsky](https://github.com/reduz) and
[Ariel Manzur](https://github.com/punto-) (both still maintaining the project) for several
years as an in-house engine, used to publish several work-for-hire titles.

![Screenshot of a 3D scene in the Godot Engine editor](https://raw.githubusercontent.com/godotengine/godot-design/master/screenshots/editor_tps_demo_1920x1080.jpg)

## Getting the engine

### Binary downloads

Official binaries for the Godot editor and the export templates can be found
[on the homepage](https://godotengine.org/download).

### Compiling from source

[See the official docs](https://docs.godotengine.org/en/latest/development/compiling/)
for compilation instructions for every supported platform.

## Community and contributing

Godot is not only an engine but an ever-growing community of users and engine
developers. The main community channels are listed [on the homepage](https://godotengine.org/community).

The best way to get in touch with the core engine developers is to join the
[Godot Contributors Chat](https://chat.godotengine.org).

To get started contributing to the project, see the [contributing guide](CONTRIBUTING.md).

## Documentation and demos

The official documentation is hosted on [ReadTheDocs](https://docs.godotengine.org).
It is maintained by the Godot community in its own [GitHub repository](https://github.com/godotengine/godot-docs).

The [class reference](https://docs.godotengine.org/en/latest/classes/)
is also accessible from the Godot editor.

We also maintain official demos in their own [GitHub repository](https://github.com/godotengine/godot-demo-projects)
as well as a list of [awesome Godot community resources](https://github.com/godotengine/awesome-godot).

There are also a number of other
[learning resources](https://docs.godotengine.org/en/latest/community/tutorials.html)
provided by the community, such as text and video tutorials, demos, etc.
Consult the [community channels](https://godotengine.org/community)
for more information.

[![Actions Build Status](https://github.com/godotengine/godot/workflows/Godot/badge.svg?branch=master)](https://github.com/godotengine/godot/actions)
[![Code Triagers Badge](https://www.codetriage.com/godotengine/godot/badges/users.svg)](https://www.codetriage.com/godotengine/godot)
[![Translate on Weblate](https://hosted.weblate.org/widgets/godot-engine/-/godot/svg-badge.svg)](https://hosted.weblate.org/engage/godot-engine/?utm_source=widget)
[![TODOs](https://badgen.net/https/api.tickgit.com/badgen/github.com/godotengine/godot)](https://www.tickgit.com/browse?repo=github.com/godotengine/godot)
