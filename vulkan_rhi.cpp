#define GLFW_INCLUDE_VULKAN
#include "vulkan_rhi.h"
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <set>

VulkanRHI::~VulkanRHI()
{
    clearSwapchain();

    vkDestroyPipeline(m_device, m_graphics_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);

    vkDestroyRenderPass(m_device, m_renderpass, nullptr);

    for (size_t i = 0; i < k_max_frames_in_flight; i++) {
        vkDestroySemaphore(m_device, m_image_available_for_render_semaphores[i], nullptr);
        vkDestroySemaphore(m_device, m_image_finished_for_presentation_semaphores[i], nullptr);
        vkDestroyFence(m_device, m_frames_in_flight_fences[i], nullptr);
    }

    vkDestroyCommandPool(m_device, m_command_pool, nullptr);

    vkDestroyDevice(m_device, nullptr);

    if (m_enable_validation_Layers) {
        destoryDebugUtilsMessengerEXT(m_instance, m_debug_messenger, nullptr);
    }

    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vkDestroyInstance(m_instance, nullptr);

    glfwDestroyWindow(m_window);

    glfwTerminate();
}

void VulkanRHI::initialize()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(WIDTH,HEIGHT,"Vulkan",nullptr,nullptr);

    //初始化Vulkan
    createInstance();
    initializeDebugMessenger();
    createWindowSurface();
    initializePhysicalDevice();
    createLogicalDevice();
    createCommandPool();
    createCommandBuffers();
    createSyncPrimitives();
    createSwapChain();
    createSwapchainImageViews();

    //创建绘制管道
    createRenderPass();
    createFramebuffer();
    createGraphicsPipeline();
}

void VulkanRHI::createInstance()
{
    if (m_enable_validation_Layers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello MyEngine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instance_create_info{};
    instance_create_info.sType  = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.pApplicationInfo = &appInfo; 
    auto extensions                              = getRequiredExtensions();
    instance_create_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    instance_create_info.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo {};
    if(m_enable_validation_Layers){
        instance_create_info.enabledLayerCount = static_cast<uint32_t>(m_validation_layers.size());
        instance_create_info.ppEnabledLayerNames = m_validation_layers.data();
        
        populateDebugMessengerCreateInfo(debugCreateInfo);

        instance_create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else{
        instance_create_info.enabledLayerCount = 0;
        instance_create_info.pNext             = nullptr;
    }

    if (vkCreateInstance(&instance_create_info, nullptr, &m_instance) != VK_SUCCESS){
            throw std::runtime_error("failed to create instance!");
        }
}

void VulkanRHI::initializeDebugMessenger(){
    if (m_enable_validation_Layers)
    {
        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);
        if (VK_SUCCESS != createDebugUtilsMessengerEXT(m_instance, &createInfo, nullptr, &m_debug_messenger))
        {
            std::runtime_error("failed to set up debug messenger!");
        }
    }
}

void VulkanRHI::createWindowSurface(){
    if(glfwCreateWindowSurface(m_instance,m_window,nullptr,&m_surface)!=VK_SUCCESS){
        throw std::runtime_error("glfwCreateWindowSurface failed!");
    }
}

void VulkanRHI::initializePhysicalDevice(){
        uint32_t physical_device_count;
        vkEnumeratePhysicalDevices(m_instance, &physical_device_count, nullptr);
        if (physical_device_count == 0)
        {
            throw std::runtime_error("enumerate physical devices failed!");
        }
        else
        {
            // find one device that matches our requirement
            // or find which is the best
            std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
            vkEnumeratePhysicalDevices(m_instance, &physical_device_count, physical_devices.data());

            std::vector<std::pair<int, VkPhysicalDevice>> ranked_physical_devices;
            for (const auto& device : physical_devices)
            {
                VkPhysicalDeviceProperties physical_device_properties;
                vkGetPhysicalDeviceProperties(device, &physical_device_properties);
                int score = 0;

                if (physical_device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                {
                    score += 1000;
                }
                else if (physical_device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
                {
                    score += 100;
                }

                ranked_physical_devices.push_back({score, device});
            }

            std::sort(ranked_physical_devices.begin(),
                      ranked_physical_devices.end(),
                      [](const std::pair<int, VkPhysicalDevice>& p1, const std::pair<int, VkPhysicalDevice>& p2) {
                          return p1 > p2;
                      });

            for (const auto& device : ranked_physical_devices)
            {
                if (isDeviceSuitable(device.second))
                {
                    m_physical_device = device.second;
                    break;
                }
            }

            if (m_physical_device == VK_NULL_HANDLE)
            {
                throw std::runtime_error("failed to find suitable physical device");
            }
        }
}

void VulkanRHI::createLogicalDevice(){
    m_queue_indices = findQueueFamilies(m_physical_device);

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos; // all queues that need to be created
    std::set<uint32_t>                   queue_families = {m_queue_indices.graphics_family.value(),
                                            m_queue_indices.present_family.value()};

    float queue_priority = 1.0f;
    //目前只设置单个队列
    for (uint32_t queue_family : queue_families) // for every queue family
    {
        // queue create info
        VkDeviceQueueCreateInfo queue_create_info {};
        queue_create_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = queue_family;
        queue_create_info.queueCount       = 1;
        queue_create_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_create_info);
    }

    // physical device features
    VkPhysicalDeviceFeatures physical_device_features = {};
    //一些物理设备的特性可选项
    // physical_device_features.samplerAnisotropy = VK_TRUE;

    // // support inefficient readback storage buffer
    // physical_device_features.fragmentStoresAndAtomics = VK_TRUE;

    // // support independent blending
    // physical_device_features.independentBlend = VK_TRUE;

    // // support geometry shader
    // if (m_enable_point_light_shadow)
    // {
    //     physical_device_features.geometryShader = VK_TRUE;
    // }

    // device create info
    VkDeviceCreateInfo device_create_info {};
    device_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pQueueCreateInfos       = queue_create_infos.data();
    device_create_info.queueCreateInfoCount    = static_cast<uint32_t>(queue_create_infos.size());
    device_create_info.pEnabledFeatures        = &physical_device_features;
    device_create_info.enabledExtensionCount   = static_cast<uint32_t>(m_device_extensions.size());
    device_create_info.ppEnabledExtensionNames = m_device_extensions.data();
    device_create_info.enabledLayerCount       = 0;

    if (vkCreateDevice(m_physical_device, &device_create_info, nullptr, &m_device) != VK_SUCCESS)
    {
        throw std::runtime_error("vk create device");
    }

    vkGetDeviceQueue(m_device, m_queue_indices.graphics_family.value(), 0, &m_graphics_queue);
    vkGetDeviceQueue(m_device, m_queue_indices.present_family.value(), 0, &m_present_queue);

    // initialize queues of this device
    // VkQueue vk_graphics_queue;
    // vkGetDeviceQueue(m_device, m_queue_indices.graphics_family.value(), 0, &vk_graphics_queue);
    // m_graphics_queue = new VulkanQueue();
    // ((VulkanQueue*)m_graphics_queue)->setResource(vk_graphics_queue);

    // vkGetDeviceQueue(m_device, m_queue_indices.present_family.value(), 0, &m_present_queue);

    // VkQueue vk_compute_queue;
    // vkGetDeviceQueue(m_device, m_queue_indices.m_compute_family.value(), 0, &vk_compute_queue);
    // m_compute_queue = new VulkanQueue();
    // ((VulkanQueue*)m_compute_queue)->setResource(vk_compute_queue);

    // more efficient pointer 有效的函数指针，但是我并不会使用！！
    // _vkResetCommandPool      = (PFN_vkResetCommandPool)vkGetDeviceProcAddr(m_device, "vkResetCommandPool");
    // _vkBeginCommandBuffer    = (PFN_vkBeginCommandBuffer)vkGetDeviceProcAddr(m_device, "vkBeginCommandBuffer");
    // _vkEndCommandBuffer      = (PFN_vkEndCommandBuffer)vkGetDeviceProcAddr(m_device, "vkEndCommandBuffer");
    // _vkCmdBeginRenderPass    = (PFN_vkCmdBeginRenderPass)vkGetDeviceProcAddr(m_device, "vkCmdBeginRenderPass");
    // _vkCmdNextSubpass        = (PFN_vkCmdNextSubpass)vkGetDeviceProcAddr(m_device, "vkCmdNextSubpass");
    // _vkCmdEndRenderPass      = (PFN_vkCmdEndRenderPass)vkGetDeviceProcAddr(m_device, "vkCmdEndRenderPass");
    // _vkCmdBindPipeline       = (PFN_vkCmdBindPipeline)vkGetDeviceProcAddr(m_device, "vkCmdBindPipeline");
    // _vkCmdSetViewport        = (PFN_vkCmdSetViewport)vkGetDeviceProcAddr(m_device, "vkCmdSetViewport");
    // _vkCmdSetScissor         = (PFN_vkCmdSetScissor)vkGetDeviceProcAddr(m_device, "vkCmdSetScissor");
    // _vkWaitForFences         = (PFN_vkWaitForFences)vkGetDeviceProcAddr(m_device, "vkWaitForFences");
    // _vkResetFences           = (PFN_vkResetFences)vkGetDeviceProcAddr(m_device, "vkResetFences");
    // _vkCmdDrawIndexed        = (PFN_vkCmdDrawIndexed)vkGetDeviceProcAddr(m_device, "vkCmdDrawIndexed");
    // _vkCmdBindVertexBuffers  = (PFN_vkCmdBindVertexBuffers)vkGetDeviceProcAddr(m_device, "vkCmdBindVertexBuffers");
    // _vkCmdBindIndexBuffer    = (PFN_vkCmdBindIndexBuffer)vkGetDeviceProcAddr(m_device, "vkCmdBindIndexBuffer");
    // _vkCmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)vkGetDeviceProcAddr(m_device, "vkCmdBindDescriptorSets");
    // _vkCmdClearAttachments   = (PFN_vkCmdClearAttachments)vkGetDeviceProcAddr(m_device, "vkCmdClearAttachments");

    // m_depth_image_format = (RHIFormat)findDepthFormat();
}

void VulkanRHI::createSwapChain(){
        // query all supports of this physical device
        SwapChainSupportDetails swapchain_support_details = querySwapChainSupport(m_physical_device);

        // choose the best or fitting format
        VkSurfaceFormatKHR chosen_surface_format =
            chooseSwapchainSurfaceFormatFromDetails(swapchain_support_details.formats);
        // choose the best or fitting present mode
        VkPresentModeKHR chosen_presentMode =
            chooseSwapchainPresentModeFromDetails(swapchain_support_details.presentModes);
        // choose the best or fitting extent
        VkExtent2D chosen_extent = chooseSwapchainExtentFromDetails(swapchain_support_details.capabilities);
        //没有理解为什么需要+1
        uint32_t image_count = swapchain_support_details.capabilities.minImageCount + 1;
        if (swapchain_support_details.capabilities.maxImageCount > 0 &&
            image_count > swapchain_support_details.capabilities.maxImageCount)
        {
            image_count = swapchain_support_details.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo {};
        createInfo.sType   = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = m_surface;

        createInfo.minImageCount    = image_count;
        createInfo.imageFormat      = chosen_surface_format.format;
        createInfo.imageColorSpace  = chosen_surface_format.colorSpace;
        createInfo.imageExtent      = chosen_extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        uint32_t queueFamilyIndices[] = {m_queue_indices.graphics_family.value(), m_queue_indices.present_family.value()};

        if (m_queue_indices.graphics_family != m_queue_indices.present_family)
        {
            //两个队列不同时需要采用并行模式来解决队列对图像所有权的问题
            createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices   = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform   = swapchain_support_details.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode    = chosen_presentMode;
        createInfo.clipped        = VK_TRUE;

        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain) != VK_SUCCESS)
        {
            throw std::runtime_error("vk create swapchain khr");
        }

        vkGetSwapchainImagesKHR(m_device, m_swapchain, &image_count, nullptr);
        m_swapchain_images.resize(image_count);
        vkGetSwapchainImagesKHR(m_device, m_swapchain, &image_count, m_swapchain_images.data());

        m_swapchain_format = chosen_surface_format.format;
        m_swapchain_extent = chosen_extent;


        //m_scissor = {{0, 0}, {m_swapchain_extent.width, m_swapchain_extent.height}};
}

void VulkanRHI::clearSwapchain(){
    for (auto framebuffer : m_swapchain_framebuffers) {
        vkDestroyFramebuffer(m_device, framebuffer, nullptr);
    }

    for (auto imageView : m_swapchain_imageviews) {
        vkDestroyImageView(m_device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
}

void VulkanRHI::recreateSwapChain()
{
    glfwGetFramebufferSize(m_window, &WIDTH, &HEIGHT);
    while (WIDTH == 0 || HEIGHT == 0) {
        glfwGetFramebufferSize(m_window, &WIDTH, &HEIGHT);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(m_device);

    clearSwapchain();
    createSwapChain();
    createSwapchainImageViews();
    createFramebuffer();
}

void VulkanRHI::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_renderpass;
    renderPassInfo.framebuffer = m_swapchain_framebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = m_swapchain_extent;

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphics_pipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) m_swapchain_extent.width;
    viewport.height = (float) m_swapchain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_swapchain_extent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}
// void VulkanRHI::clearSwapchain(){
//         // for (auto framebuffer : swapChainFramebuffers) {
//         //     vkDestroyFramebuffer(device, framebuffer, nullptr);
//         // }

//         // for (auto imageView : swapChainImageViews) {
//         //     vkDestroyImageView(device, imageView, nullptr);
//         // }

//         // vkDestroySwapchainKHR(device, swapChain, nullptr);
// }

void VulkanRHI::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    //Allow command buffers to be rerecorded individually, without this flag they all have to be reset together
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_queue_indices.graphics_family.value();

    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_command_pool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }    
}

void VulkanRHI::createCommandBuffers()
{
    m_command_buffers.resize(k_max_frames_in_flight);

    VkCommandBufferAllocateInfo command_buffer_allocate_info {};
    command_buffer_allocate_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_allocate_info.commandBufferCount = (uint32_t) m_command_buffers.size();

    if (vkAllocateCommandBuffers(m_device, &command_buffer_allocate_info, m_command_buffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void VulkanRHI::createDescriptorPool()
{
    
}

void VulkanRHI::createSyncPrimitives()
{
    m_image_available_for_render_semaphores.resize(k_max_frames_in_flight);
    m_image_finished_for_presentation_semaphores.resize(k_max_frames_in_flight);
    
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < k_max_frames_in_flight; i++) {
        if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_image_available_for_render_semaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_image_finished_for_presentation_semaphores[i]) != VK_SUCCESS ||
            vkCreateFence(m_device, &fenceInfo, nullptr, &m_frames_in_flight_fences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}

void VulkanRHI::createSwapchainImageViews()
{
    m_swapchain_imageviews.resize(k_max_frames_in_flight);
    for (size_t i = 0; i < m_swapchain_imageviews.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = m_swapchain_images[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = m_swapchain_format;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_device, &createInfo, nullptr, &m_swapchain_imageviews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }
}

void VulkanRHI::createRenderPass(){
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = m_swapchain_format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderpass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

void VulkanRHI::createGraphicsPipeline(){
    auto vertShaderCode = readFile("shaders/vert.spv");
    auto fragShaderCode = readFile("shaders/frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipeline_layout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = m_pipeline_layout;
    pipelineInfo.renderPass = m_renderpass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphics_pipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, vertShaderModule, nullptr);
}

void VulkanRHI::createFramebuffer()
{
    m_swapchain_framebuffers.resize(m_swapchain_imageviews.size());

    for (size_t i = 0; i < m_swapchain_imageviews.size(); i++) {
        VkImageView attachments[] = {
            m_swapchain_imageviews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = m_renderpass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width  =    m_swapchain_extent.width;
        framebufferInfo.height =    m_swapchain_extent.height;
        framebufferInfo.layers =    1;

        if (vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_swapchain_framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

VkShaderModule VulkanRHI::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                                    VkDebugUtilsMessageTypeFlagsEXT,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                    void*)
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

void VulkanRHI::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
        createInfo       = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
}

void VulkanRHI::destoryDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) 
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

bool VulkanRHI::checkValidationLayerSupport(){
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : m_validation_layers)
    {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

bool VulkanRHI::isDeviceSuitable(VkPhysicalDevice physicalm_device)
{
    auto queue_indices           = findQueueFamilies(physicalm_device);
    //目前只检查了显卡是否能够进行显示，这边描述不够清晰
    bool is_extensions_supported = checkDeviceExtensionSupport(physicalm_device);
    bool is_swapchain_adequate   = false;
    if (is_extensions_supported)
    {
        SwapChainSupportDetails swapchain_support_details = querySwapChainSupport(physicalm_device);
        is_swapchain_adequate =
            !swapchain_support_details.formats.empty() && !swapchain_support_details.presentModes.empty();
    }

    //VkPhysicalDeviceFeatures physicalm_device_features;
    //vkGetPhysicalDeviceFeatures(physicalm_device, &physicalm_device_features);

    if (!queue_indices.isComplete() || !is_swapchain_adequate)
    {
        return false;
    }

    return true;
}

QueueFamilyIndices VulkanRHI::findQueueFamilies(VkPhysicalDevice physicalm_device) // for device and surface
{
        QueueFamilyIndices indices;
        uint32_t           queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalm_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalm_device, &queue_family_count, queue_families.data());

        int i = 0;
        for (const auto& queue_family : queue_families)
        {
            if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) // if support graphics command queue
            {
                indices.graphics_family = i;
            }

            VkBool32 is_present_support = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalm_device,
                                                 i,
                                                 m_surface,
                                                 &is_present_support); // if support surface presentation
            if (is_present_support)
            {
                indices.present_family = i;
            }

            if (indices.isComplete())
            {
                break;
            }
            i++;
        }
        return indices;
    }

bool VulkanRHI::checkDeviceExtensionSupport(VkPhysicalDevice physicalm_device){
        uint32_t extension_count;
        vkEnumerateDeviceExtensionProperties(physicalm_device, nullptr, &extension_count, nullptr);

        std::vector<VkExtensionProperties> available_extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(physicalm_device, nullptr, &extension_count, available_extensions.data());

        std::set<std::string> required_extensions(m_device_extensions.begin(), m_device_extensions.end());
        for (const auto& extension : available_extensions)
        {
            required_extensions.erase(extension.extensionName);
        }

        return required_extensions.empty();
}

SwapChainSupportDetails VulkanRHI::querySwapChainSupport(VkPhysicalDevice physicalm_device){
     SwapChainSupportDetails details_result;

        // capabilities(min/max number of images in swap chain, min/max width and height of images)
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalm_device, m_surface, &details_result.capabilities);

        // formats
        uint32_t format_count;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalm_device, m_surface, &format_count, nullptr);
        if (format_count != 0)
        {
            details_result.formats.resize(format_count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                physicalm_device, m_surface, &format_count, details_result.formats.data());
        }

        // present modes
        uint32_t presentmode_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalm_device, m_surface, &presentmode_count, nullptr);
        if (presentmode_count != 0)
        {
            details_result.presentModes.resize(presentmode_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                physicalm_device, m_surface, &presentmode_count, details_result.presentModes.data());
        }

        return details_result;
}

VkSurfaceFormatKHR VulkanRHI::chooseSwapchainSurfaceFormatFromDetails(const std::vector<VkSurfaceFormatKHR>& available_surface_formats)
{
    for (const auto& surface_format : available_surface_formats)
    {
        // TODO: select the VK_FORMAT_B8G8R8A8_SRGB surface format,
        // there is no need to do gamma correction in the fragment shader
        if (surface_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            surface_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return surface_format;
        }
    }
    return available_surface_formats[0];
}

VkPresentModeKHR VulkanRHI::chooseSwapchainPresentModeFromDetails(const std::vector<VkPresentModeKHR>& available_present_modes)
{
    for (VkPresentModeKHR present_mode : available_present_modes)
    {
        if (VK_PRESENT_MODE_MAILBOX_KHR == present_mode)
        {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanRHI::chooseSwapchainExtentFromDetails(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX)
    {
        return capabilities.currentExtent;
    }
    else
    {
        int width, height;
        glfwGetFramebufferSize(m_window, &width, &height);

        VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

        actualExtent.width =
            std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height =
            std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

void VulkanRHI::draw()
{
    //current
    vkWaitForFences(m_device, 1, &m_frames_in_flight_fences[m_current_frame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(m_device, 
                                            m_swapchain, 
                                            UINT64_MAX, 
                                            m_image_finished_for_presentation_semaphores[m_current_frame], 
                                            VK_NULL_HANDLE, 
                                            &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    vkResetFences(m_device,1,&m_frames_in_flight_fences[m_current_frame]);

    vkResetCommandBuffer(m_command_buffers[m_current_frame], /*VkCommandBufferResetFlagBits*/ 0);
    recordCommandBuffer(m_command_buffers[m_current_frame], imageIndex);
}




