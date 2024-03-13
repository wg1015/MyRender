#pragma once

#include <memory>




class RenderSystem
{
private:
    std::shared_ptr<RHI> m_rhi;
public:
    void initialize(RenderSystemInitInfo init_info);
    RenderSystem(/* args */);
    ~RenderSystem();
};


