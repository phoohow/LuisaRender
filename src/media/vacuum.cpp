//
// Created by ChenXin on 2023/3/8.
//

#include <base/medium.h>
#include <base/pipeline.h>

namespace luisa::render {

using compute::Ray;

class VacuumMedium : public Medium {

public:
    class VacuumMajorantIterator : public RayMajorantIterator {
    private:
        uint _dim;

    public:
        explicit VacuumMajorantIterator(uint dim) noexcept : _dim(dim) {}
        [[nodiscard]] RayMajorantSegment next() noexcept override {
            return RayMajorantSegment::one(_dim);
        }
    };

    class VacuumMediumInstance;

    class VacuumMediumClosure : public Medium::Closure {
    public:
        [[nodiscard]] Medium::Sample sample(Expr<float> t_max, PCG32 &rng) const noexcept override {
            return Medium::Sample::zero(swl().dimension());
        }
        [[nodiscard]] Evaluation transmittance(Expr<float> t, PCG32 &rng) const noexcept override {
            return Medium::Evaluation::zero(swl().dimension());
        }
        [[nodiscard]] unique_ptr<RayMajorantIterator> sample_iterator(Expr<float> t_max) const noexcept override {
            return luisa::make_unique<VacuumMajorantIterator>(swl().dimension());
        }

    public:
        VacuumMediumClosure(
            const VacuumMediumInstance *instance, Expr<Ray> ray,
            const SampledWavelengths &swl, Expr<float> time) noexcept
            : Medium::Closure{instance, ray, swl, time, 1.0f,
                              SampledSpectrum{swl.dimension(), 0.f}, SampledSpectrum{swl.dimension(), 0.f},
                              SampledSpectrum{swl.dimension(), 0.f}, nullptr} {}
    };

    class VacuumMediumInstance : public Medium::Instance {

    private:
        friend class VacuumMedium;

    public:
        VacuumMediumInstance(const Pipeline &pipeline, const Medium *medium) noexcept
            : Medium::Instance(pipeline, medium) {}
        [[nodiscard]] luisa::unique_ptr<Closure> closure(
            Expr<Ray> ray, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
            return luisa::make_unique<VacuumMediumClosure>(this, ray, swl, time);
        }
    };

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<VacuumMediumInstance>(pipeline, this);
    }

public:
    VacuumMedium(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Medium{scene, desc} {
        _priority = VACUUM_PRIORITY;
    }
    [[nodiscard]] bool is_vacuum() const noexcept override { return true; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::VacuumMedium)