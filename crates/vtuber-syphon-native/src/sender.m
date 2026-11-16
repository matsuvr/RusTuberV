// RusTuberV MIT-licensed bridge. API signatures follow SyphonMetalServer.h:
// https://github.com/syphon/syphon-framework/blob/f4761677a45b8034a3c2069ec0f3d2553da81fba/SyphonMetalServer.h
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>

// Dynamically load only the bundled framework, so an absent framework is a
// recoverable output error rather than an application-launch linker failure.
@protocol RTVSyphonServer <NSObject>
- (id)initWithName:(NSString *)name device:(id<MTLDevice>)device options:(NSDictionary *)options;
- (void)publishFrameTexture:(id<MTLTexture>)texture onCommandBuffer:(id<MTLCommandBuffer>)buffer imageRegion:(NSRect)region flipped:(BOOL)flipped;
- (void)stop;
@end
@interface RTVSyphonSender : NSObject
@property(strong) id<RTVSyphonServer> server;
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> queue;
@property(strong) id<MTLTexture> texture;
@end
@implementation RTVSyphonSender
@end

int32_t rtv_syphon_create(void **out) {
    @autoreleasepool {
        *out = NULL;
        NSBundle *main = [NSBundle mainBundle];
        NSString *path;
        if ([[main bundlePath] hasSuffix:@".app"]) {
            path = [[main privateFrameworksPath] stringByAppendingPathComponent:@"Syphon.framework"];
        } else {
            path = [[[main executablePath] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"Syphon.framework"];
        }
        NSBundle *framework = [NSBundle bundleWithPath:path];
        NSError *error = nil;
        if (!framework || ![framework loadAndReturnError:&error]) return 1;
        Class serverClass = [framework classNamed:@"SyphonMetalServer"];
        if (!serverClass) return 2;
        RTVSyphonSender *sender = [RTVSyphonSender new];
        sender.device = MTLCreateSystemDefaultDevice();
        sender.queue = [sender.device newCommandQueue];
        if (!sender.device || !sender.queue) return 3;
        sender.server = [(id<RTVSyphonServer>)[serverClass alloc] initWithName:@"RusTuberV" device:sender.device options:nil];
        if (!sender.device || !sender.queue || !sender.server) return 3;
        *out = (void *)CFBridgingRetain(sender);
        return 0;
    }
}

int32_t rtv_syphon_send(void *handle, const uint8_t *bytes, uint32_t width, uint32_t height) {
    @autoreleasepool {
        RTVSyphonSender *sender = (__bridge RTVSyphonSender *)handle;
        if (!sender.texture || sender.texture.width != width || sender.texture.height != height) {
            MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:width height:height mipmapped:NO];
            desc.storageMode = MTLStorageModeShared;
            desc.usage = MTLTextureUsageShaderRead;
            sender.texture = [sender.device newTextureWithDescriptor:desc];
        }
        if (!sender.texture) return 1;
        [sender.texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 withBytes:bytes bytesPerRow:(NSUInteger)width * 4];
        id<MTLCommandBuffer> buffer = [sender.queue commandBuffer];
        if (!buffer) return 1;
        [sender.server publishFrameTexture:sender.texture onCommandBuffer:buffer imageRegion:NSMakeRect(0, 0, width, height) flipped:NO];
        [buffer commit];
        // One in-flight GPU command prevents unbounded buffering and reuse of the
        // source texture while Syphon is still copying. Only this worker waits.
        [buffer waitUntilCompleted];
        return buffer.status == MTLCommandBufferStatusCompleted ? 0 : 1;
    }
}

void rtv_syphon_destroy(void *handle) {
    @autoreleasepool {
        RTVSyphonSender *sender = CFBridgingRelease(handle);
        [sender.server stop];
    }
}
