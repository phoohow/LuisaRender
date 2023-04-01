#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/texture.h>
#include <base/scene.h>
#include <base/pipeline.h>

#include <utility>

namespace luisa::render {

class vrayMtlSurface : public Surface {

private:
    const Texture *_color{};
    const Texture *_roughness{};
    bool _remap_roughness;

public:
    vrayMtlSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _color{scene->load_texture(desc->property_node_or_default(
              "color", lazy_construct([desc] {
                  return desc->property_node_or_default("Kd");
              })))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
#define LUISA_RENDER_vrayMtl_PARAM_LOAD(name) \
    _##name = scene->load_texture(desc->property_node_or_default(#name));
        LUISA_RENDER_vrayMtl_PARAM_LOAD(roughness)
#undef LUISA_RENDER_vrayMtl_PARAM_LOAD
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override {
        return LUISA_RENDER_PLUGIN_NAME;
    }
    [[nodiscard]] uint properties() const noexcept override {
        auto properties = property_reflective;
        // TODO:
        return properties;
    }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class vrayMtlSurfaceInstance : public Surface::Instance {

private:
    const Texture::Instance *_color{};
    const Texture::Instance *_roughness{};

public:
    vrayMtlSurfaceInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *color, const Texture::Instance *roughness) noexcept
        : Surface::Instance{pipeline, surface},
          _color{color}, _roughness{roughness} {}
    [[nodiscard]] auto color() const noexcept { return _color; }
    [[nodiscard]] auto roughness() const noexcept { return _roughness; }

public:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
        Expr<float3> wo, Expr<float> eta, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> vrayMtlSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto color = pipeline.build_texture(command_buffer, _color);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    return luisa::make_unique<vrayMtlSurfaceInstance>(
        pipeline, this, color, roughness);
}

using namespace compute;

namespace {

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
//
// The Schlick Fresnel approximation is:
//
// R = R(0) + (1 - R(0)) (1 - cos theta)^5,
//
// where R(0) is the reflectance at normal indicence.
[[nodiscard]] inline Float SchlickWeight(Expr<float> cosTheta) noexcept {
    auto m = saturate(1.f - cosTheta);
    return sqr(sqr(m)) * m;
}

[[nodiscard]] inline Float FrSchlick(Expr<float> R0, Expr<float> cosTheta) noexcept {
    return lerp(R0, 1.f, SchlickWeight(cosTheta));
}

// For a dielectric, R(0) = (eta - 1)^2 / (eta + 1)^2, assuming we're
// coming from air..
[[nodiscard]] inline Float SchlickR0FromEta(Float eta) {
    return sqr((eta - 1.f) / (eta + 1.f));
}

class vrayMtlSheen final : public BxDF {

private:
    SampledSpectrum R;

public:
    explicit vrayMtlSheen(const SampledSpectrum &R) noexcept : R{R} {}
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return R; }
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        static Callable impl = [](Float3 wo, Float3 wi) noexcept {
            auto wh = wi + wo;
            auto valid = any(wh != 0.f);
            wh = normalize(wh);
            auto cosThetaD = dot(wi, wh);
            return ite(valid, SchlickWeight(cosThetaD), 0.f);
        };
        return R * impl(wo, wi);
    }
};

[[nodiscard]] inline Float GTR1(Float cosTheta, Float alpha) noexcept {
    auto alpha2 = sqr(alpha);
    auto denom = pi * log(alpha2) * (1.f + (alpha2 - 1.f) * sqr(cosTheta));
    return (alpha2 - 1.f) / denom;
}

// Smith masking/shadowing term.
[[nodiscard]] inline Float smithG_GGX(Float cosTheta, Float alpha) noexcept {
    auto alpha2 = sqr(alpha);
    auto cosTheta2 = sqr(cosTheta);
    return 1.f / (cosTheta + sqrt(alpha2 + cosTheta2 - alpha2 * cosTheta2));
}

class vrayMtlClearcoat final {

private:
    Float weight;
    Float gloss;

public:
    vrayMtlClearcoat(Float weight, Float gloss) noexcept
        : weight{std::move(weight)}, gloss{std::move(gloss)} {}
    [[nodiscard]] Float evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept {
        static Callable impl = [](Float3 wo, Float3 wi, Float weight, Float gloss) noexcept {
            auto wh = wi + wo;
            auto valid = any(wh != 0.f);
            wh = normalize(wh);
            // Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
            // GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
            // (which is GTR2).
            auto Dr = GTR1(abs_cos_theta(wh), gloss);
            auto Fr = FrSchlick(.04f, dot(wo, wh));
            // The geometric term always based on alpha = 0.25.
            auto Gr = smithG_GGX(abs_cos_theta(wo), .25f) *
                      smithG_GGX(abs_cos_theta(wi), .25f);
            return ite(valid, weight * Gr * Fr * Dr * .25f, 0.f);
        };
        return impl(wo, wi, weight, gloss);
    }
    [[nodiscard]] BxDF::SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u) const noexcept {
        static Callable impl = [](Float3 wo, Float2 u, Float gloss) noexcept {
            // TODO: double check all this: there still seem to be some very
            // occasional fireflies with clearcoat; presumably there is a bug
            // somewhere.
            auto alpha2 = gloss * gloss;
            auto cosTheta = sqrt(max(0.f, (1.f - pow(alpha2, 1.f - u[0])) / (1.f - alpha2)));
            auto sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
            auto phi = 2.f * pi * u[1];
            auto wh = spherical_direction(sinTheta, cosTheta, phi);
            wh = ite(same_hemisphere(wo, wh), wh, -wh);
            return reflect(wo, wh);
        };
        auto wi = impl(wo, u, gloss);
        return {.wi = wi, .valid = same_hemisphere(wo, wi)};
    }
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept {
        static Callable impl = [](Float3 wo, Float3 wi, Float gloss) noexcept {
            auto wh = wi + wo;
            auto valid = same_hemisphere(wo, wi) & any(wh != 0.f);
            wh = normalize(wh);
            // The sampling routine samples wh exactly from the GTR1 distribution.
            // Thus, the final value of the PDF is just the value of the
            // distribution for wh converted to a mesure with respect to the
            // surface normal.
            auto Dr = GTR1(abs_cos_theta(wh), gloss);
            return ite(valid, Dr * abs_cos_theta(wh) / (4.f * dot(wo, wh)), 0.f);
        };
        return impl(wo, wi, gloss);
    }
};

// Specialized Fresnel function used for the specular component, based on
// a mixture between dielectric and the Schlick Fresnel approximation.
class vrayMtlFresnel final : public Fresnel {

private:
    SampledSpectrum R0;
    Float metallic;
    Float e;
    bool two_sided;

public:
    vrayMtlFresnel(const SampledSpectrum &R0, Float metallic, Float eta, bool two_sided) noexcept
        : R0{R0}, metallic{std::move(metallic)}, e{std::move(eta)}, two_sided{two_sided} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float> cosI_in) const noexcept override {
        auto cosI = two_sided ? abs(cosI_in) : def(cosI_in);
        auto fr = fresnel_dielectric(cosI, 1.f, e);
        auto f0 = R0.map([cosI](auto x) noexcept { return FrSchlick(x, cosI); });
        return lerp(fr, f0, metallic);
    }
    [[nodiscard]] auto &eta() const noexcept { return e; }
};

struct vrayMtlMicrofacetDistribution final : public TrowbridgeReitzDistribution {
    explicit vrayMtlMicrofacetDistribution(Expr<float2> alpha) noexcept
        : TrowbridgeReitzDistribution{alpha} {}
};

}// namespace

class vrayMtlSurfaceClosureImpl {
public:
    static constexpr auto max_sampling_techique_count = 4u;

private:
    SampledSpectrum _color;
    luisa::unique_ptr<BxDF> _diffuse;
    luisa::unique_ptr<vrayMtlMicrofacetDistribution> _distrib;
    Float _sampling_weights[max_sampling_techique_count];
    uint _sampling_technique_count{0u};
    uint _diffuse_like_technique_index{~0u};

private:
    [[nodiscard]] Surface::Evaluation _evaluate_local(const Surface::Closure *cls,
                                                      Expr<float3> wo_local, Expr<float3> wi_local,
                                                      TransportMode mode) const noexcept {
        SampledSpectrum f{cls->swl().dimension(), 0.f};
        auto pdf = def(0.f);
        $if(same_hemisphere(wo_local, wi_local)) {// reflection
            if (_diffuse) {
                f += _diffuse->evaluate(wo_local, wi_local, mode);
                pdf += _sampling_weights[_diffuse_like_technique_index] *
                       _diffuse->pdf(wo_local, wi_local, mode);
            }

            // TODO: Add support for sheen, specular, clearcoat lobes
        }
        $else{ // transmission
            // TODO: Add support for transmission
        };
        return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    }

public:
    [[nodiscard]] SampledSpectrum albedo() const noexcept { return _color; }
    [[nodiscard]] Float2 roughness() const noexcept {
        return vrayMtlMicrofacetDistribution::alpha_to_roughness(_distrib->alpha());
    }
    vrayMtlSurfaceClosureImpl(const Surface::Closure *cls, const Texture::Instance *color_tex,
                              const Texture::Instance *roughness_tex) noexcept
        : _color{cls->swl().dimension()} {
        auto color_decode = color_tex ?
                                color_tex->evaluate_albedo_spectrum(*cls->it(), cls->swl(), cls->time()) :
                                Spectrum::Decode::one(cls->swl().dimension());
        _color = color_decode.value;
        auto color_lum = color_decode.strength;
        auto diffuse_weight = 1.0f;
        auto roughness = roughness_tex ? roughness_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;

        // TODO: have no idea what is about remap roughness
        // if (cls->instance()->node<vrayMtlSurface>()->remap_roughness()) {
        //     roughness = vrayMtlMicrofacetDistribution::roughness_to_alpha(roughness);
        // }

        auto tint_weight = ite(color_lum > 0.f, 1.f / color_lum, 1.f);
        auto tint = saturate(_color * tint_weight);// normalize lum. to isolate hue+sat
        auto tint_lum = color_lum * tint_weight;

        // TODO: Learn this new c++ grammer
        static constexpr auto epsilon = 1e-6f;

        // diffuse-like lobes: diffuse (TODO: other lobes dim to diffuse)
        if (color_tex == nullptr || !color_tex->node()->is_black()) {
            auto Cdiff_weight = diffuse_weight;
            auto Cdiff = _color * Cdiff_weight;
            auto useLambertian = roughness < epsilon;
            $if(useLambertian) {
                _diffuse = luisa::make_unique<LambertianReflection>(Cdiff);
            }
            $else {
                // TODO: sigma seems to be the roughness, but need to be verified next
                _diffuse = luisa::make_unique<OrenNayar>(Cdiff, roughness);
            };

            auto sampling_weight = diffuse_weight * color_lum;

            _diffuse_like_technique_index = _sampling_technique_count++;
            _sampling_weights[_diffuse_like_technique_index] = saturate(sampling_weight);
        }

        // TODO: add support for sheen, specular, clearcoat, and transmission

        // TODO: normalize sampling weights
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < _sampling_technique_count; i++) {
            sum_weights += _sampling_weights[i];
        }
        auto inv_sum_weights = ite(sum_weights == 0.f, 0.f, 1.f / sum_weights);
        for (auto &s : _sampling_weights) { s *= inv_sum_weights; }
    }

    [[nodiscard]] Surface::Evaluation evaluate(const Surface::Closure *cls,
                                               Expr<float3> wo, Expr<float3> wi,
                                               TransportMode mode) const noexcept {
        auto wo_local = cls->it()->shading().world_to_local(wo);
        auto wi_local = cls->it()->shading().world_to_local(wi);
        return _evaluate_local(cls, wo_local, wi_local, mode);
    }

    [[nodiscard]] Surface::Sample sample(const Surface::Closure *cls,
                                         Expr<float3> wo, Expr<float> u_lobe,
                                         Expr<float2> u, TransportMode mode) const noexcept {
        auto sampling_tech = def(0u);
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < _sampling_technique_count; i++) {
            sampling_tech = ite(u_lobe > sum_weights, i, sampling_tech);
            sum_weights += _sampling_weights[i];
        }
        // sample
        auto wo_local = cls->it()->shading().world_to_local(wo);
        auto event = def(Surface::event_reflect);
        BxDF::SampledDirection wi_sample;
        $switch(sampling_tech) {
            if (_diffuse) {
                $case(_diffuse_like_technique_index) {
                    wi_sample = _diffuse->sample_wi(wo_local, u, mode);
                };
            }
            // TODO: Add support for sheen, specular, clearcoat, and transmission
            $default { unreachable(); };
        };
        auto eval = Surface::Evaluation::zero(_color.dimension());
        auto wi = cls->it()->shading().local_to_world(wi_sample.wi);
        $if(wi_sample.valid) {
            eval = _evaluate_local(cls, wo_local, wi_sample.wi, mode);
        };
        return {.eval = eval, .wi = wi, .event = event};
    }
};

class vrayMtlSurfaceClosure : public Surface::Closure {

private:
    luisa::unique_ptr<vrayMtlSurfaceClosureImpl> _impl;

private:
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _impl->albedo(); }
    [[nodiscard]] Float2 roughness() const noexcept override { return _impl->roughness(); }

    [[nodiscard]] Surface::Evaluation _evaluate(Expr<float3> wo, Expr<float3> wi,
                                                TransportMode mode) const noexcept override {
        return _impl->evaluate(this, wo, wi, mode);
    }

    [[nodiscard]] Surface::Sample _sample(Expr<float3> wo, Expr<float> u_lobe,
                                          Expr<float2> u, TransportMode mode) const noexcept override {
        return _impl->sample(this, wo, u_lobe, u, mode);
    }

public:
    vrayMtlSurfaceClosure(const vrayMtlSurfaceInstance *instance, luisa::shared_ptr<Interaction> it,
                          const SampledWavelengths &swl, Expr<float> eta_i, Expr<float> time,
                          const Texture::Instance *color_tex, const Texture::Instance *roughness_tex) noexcept
        : Surface::Closure{instance, std::move(it), swl, time} {
        {
            _impl = luisa::make_unique<vrayMtlSurfaceClosureImpl>(
                this, color_tex, roughness_tex);
        }
    }
};

luisa::unique_ptr<Surface::Closure> vrayMtlSurfaceInstance::closure(
    luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
    Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept {
    return luisa::make_unique<vrayMtlSurfaceClosure>(
        this, std::move(it), swl, eta_i, time,
        _color, _roughness);
}

using NormalMapOpacityvrayMtlSurface = NormalMapWrapper<OpacitySurfaceWrapper<
    vrayMtlSurface, vrayMtlSurfaceInstance, vrayMtlSurfaceClosure>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMapOpacityvrayMtlSurface)
