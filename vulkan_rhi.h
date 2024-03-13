#define GLFW_INCLUDE_VULKAN
#include "rhi.h"
#include <GLFW/glfw3.h>
#include <vector>
#include <optional>





struct QueueFamilyIndices
{
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> present_family;

    bool isComplete() { 
        return graphics_family.has_value() && present_family.has_value();
    }
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

class VulkanRHI final : public RHI
{
public:
    virtual void initialize() final override;
    //TODO:分离draw
    void draw();
    
    static uint8_t const k_max_frames_in_flight {2};
    int WIDTH = 800U;
    int HEIGHT = 600U;
    
    virtual ~VulkanRHI() final override;
private:
    uint32_t m_current_frame {0};   

    GLFWwindow* m_window {nullptr};

    VkInstance         m_instance {nullptr};
    VkSurfaceKHR       m_surface {nullptr};
    VkPhysicalDevice   m_physical_device {nullptr};
    VkDevice           m_device {nullptr};
    VkQueue            m_present_queue {nullptr};
    VkQueue            m_graphics_queue{nullptr};
    VkRenderPass       m_renderpass{nullptr};
    VkPipelineLayout   m_pipeline_layout{nullptr};
    VkPipeline         m_graphics_pipeline{nullptr};
    
    VkSwapchainKHR              m_swapchain {nullptr};
    VkExtent2D                  m_swapchain_extent;
    VkFormat                    m_swapchain_format;
    std::vector<VkImage>        m_swapchain_images;
    std::vector<VkImageView>    m_swapchain_imageviews;
    std::vector<VkFramebuffer>  m_swapchain_framebuffers;
                                                                           
    std::vector<VkSemaphore>    m_image_available_for_render_semaphores;
    std::vector<VkSemaphore>    m_image_finished_for_presentation_semaphores;
    std::vector<VkFence>        m_frames_in_flight_fences;

    VkCommandPool      m_command_pool;
    std::vector<VkCommandBuffer> m_command_buffers;
    
    QueueFamilyIndices m_queue_indices;
    VkDebugUtilsMessengerEXT m_debug_messenger;
    
    const std::vector<char const*> m_validation_layers {"VK_LAYER_KHRONOS_validation"};
    std::vector<char const*> m_device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    bool m_enable_validation_Layers{ true };
    
    void createInstance();
    void initializeDebugMessenger();
    void createWindowSurface();
    void initializePhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
    void createCommandBuffers();
    void createDescriptorPool();
    void createSyncPrimitives();
    void createSwapchainImageViews();
    void createFramebuffer();

    void createRenderPass();
    void createGraphicsPipeline();
    
    //void createAssetAllocator();
    void createSwapChain();
    void clearSwapchain();
    void recreateSwapChain();

    VkShaderModule createShaderModule(const std::vector<char>& code);
    
    //之后找时间看看能不能分离
    //bool beginCommandBuffer();
    //bool endCommandBuffer();
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);


    std::vector<const char*> getRequiredExtensions();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
    bool checkValidationLayerSupport();
    
    VkResult createDebugUtilsMessengerEXT(VkInstance                               instance,
                                        const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                        const VkAllocationCallbacks*              pAllocator,
                                        VkDebugUtilsMessengerEXT*                 pDebugMessenger);
    void destoryDebugUtilsMessengerEXT(VkInstance                                  instance, 
                                        VkDebugUtilsMessengerEXT                    debugMessenger, 
                                        const VkAllocationCallbacks*                pAllocator);
    bool isDeviceSuitable(VkPhysicalDevice physicalm_device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalm_device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice physical_device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice physical_device);
    VkSurfaceFormatKHR chooseSwapchainSurfaceFormatFromDetails(const std::vector<VkSurfaceFormatKHR>& available_surface_formats);
    VkPresentModeKHR chooseSwapchainPresentModeFromDetails(const std::vector<VkPresentModeKHR>& available_present_modes);
    VkExtent2D chooseSwapchainExtentFromDetails(const VkSurfaceCapabilitiesKHR& capabilities);
};