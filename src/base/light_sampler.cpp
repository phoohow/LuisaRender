//
// Created by Mike Smith on 2022/1/10.
//

#include <base/light_sampler.h>
#include <base/pipeline.h>
#include <util/sampling.h>

namespace luisa::render {

LightSampler::LightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::LIGHT_SAMPLER} {}

LightSampler::Sample LightSampler::Instance::sample_selection(
    const Interaction &it_from, const Selection &sel, Expr<float2> u,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto sample = Sample::zero(swl.dimension());
    if (!pipeline().has_lighting()) { return sample; }
    if (_pipeline.environment() != nullptr) {// possibly environment lighting
        if (_pipeline.lights().empty()) {    // no lights, just environment lighting
            sample = sample_environment(it_from, sel, u, swl, time);
        } else {// environment lighting and lights
            $if(sel.tag == selection_environment) {
                sample = sample_environment(it_from, sel, u, swl, time);
            }
            $else {
                sample = sample_light(it_from, sel, u, swl, time);
            };
        }
    } else {// no environment lighting, just lights
        sample = sample_light(it_from, sel, u, swl, time);
    }
    return sample;
}
LightSampler::Sample LightSampler::Instance::sample_selection_le(
    const Selection &sel, Expr<float2> u_light, Expr<float2> u_direction, 
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto sample = Sample::zero(swl.dimension());
    if (!pipeline().has_lighting()) { return sample; }
    if (_pipeline.environment() != nullptr) {// possibly environment lighting
        if (_pipeline.lights().empty()) {    // no lights, just environment lighting
            sample = sample_environment_le(sel, u_light,u_direction, swl, time);
        } else {// environment lighting and lights
            $if(sel.tag == selection_environment) {
                sample = sample_environment_le(sel, u_light, u_direction, swl, time);
            }
            $else {
                sample = sample_light_le(sel, u_light, u_direction, swl, time);
            };
        }
    } else {// no environment lighting, just lights
        sample = sample_light_le(sel, u_light, u_direction, swl, time);
    }
    return sample;
}

LightSampler::Sample LightSampler::Instance::sample(
    const Interaction &it_from, Expr<float> u_sel, Expr<float2> u_light,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    if (!_pipeline.has_lighting()) { return Sample::zero(swl.dimension()); }
    auto sel = select(it_from, u_sel, swl, time);
    return sample_selection(it_from, sel, u_light, swl, time);
}
LightSampler::Sample LightSampler::Instance::sample_le(
    Expr<float> u_sel, Expr<float2> u_light, Expr<float2> u_direction,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    if (!_pipeline.has_lighting()) { return Sample::zero(swl.dimension()); }
    auto sel = select(u_sel, swl, time);
    return sample_selection_le(sel, u_light, u_direction, swl, time);
}

LightSampler::Sample LightSampler::Instance::sample_light(
    const Interaction &it_from, const LightSampler::Selection &sel, Expr<float2> u,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto s = _sample_light(it_from, sel.tag, u, swl, time);
    s.eval.pdf *= sel.prob;
    return Sample::from_light(s, it_from);
}

LightSampler::Sample LightSampler::Instance::sample_environment(
    const Interaction &it_from, const LightSampler::Selection &sel, Expr<float2> u,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto s = _sample_environment(u, swl, time);
    s.eval.pdf *= sel.prob;
    return Sample::from_environment(s, it_from);
}
LightSampler::Sample LightSampler::Instance::sample_light_le(
    const LightSampler::Selection &sel, Expr<float2> u_light, Expr<float2> u_direction,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto s = _sample_light_le(sel.tag, u_light,u_direction, swl, time);
    s.eval.pdf *= sel.prob;
    return s;
}

LightSampler::Sample LightSampler::Instance::sample_environment_le(
    const LightSampler::Selection &sel, Expr<float2> u_light, Expr<float2>u_direction,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto s = _sample_environment(u_direction, swl, time);
    s.eval.pdf *= sel.prob;
    auto cd = sample_uniform_disk_concentric(u_light);
    const Float WorldR = 1000.0f;
    Frame fr = Frame::make(s.wi);
    auto origin =WorldR * (fr.local_to_world(make_float3(cd.x,cd.y,-1.0f)));
    s.eval.pdf *= 1 / (pi * WorldR * WorldR);
    return Sample{.eval = s.eval, .shadow_ray = make_ray(origin,s.wi)};
}

LightSampler::Sample LightSampler::Sample::zero(uint spec_dim) noexcept {
    return Sample{.eval = Evaluation::zero(spec_dim), .shadow_ray = {}};
}

LightSampler::Sample LightSampler::Sample::from_light(const Light::Sample &s,
                                                      const Interaction &it_from) noexcept {
    return Sample{.eval = s.eval, .shadow_ray = it_from.spawn_ray_to(s.p)};
}

LightSampler::Sample LightSampler::Sample::from_environment(const Environment::Sample &s,
                                                            const Interaction &it_from) noexcept {
    return Sample{.eval = s.eval, .shadow_ray = it_from.spawn_ray(s.wi)};
}

}// namespace luisa::render
