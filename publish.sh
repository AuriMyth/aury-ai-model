#!/bin/bash

# AuriMyth Foundation Kit - PyPI 发布脚本（使用 uv）
#
# 使用方法:
#   ./publish.sh [test|prod]
#
# 参数说明:
#   test: 发布到测试 PyPI (https://test.pypi.org)
#   prod: 发布到正式 PyPI (https://pypi.org) - 默认
#
# 前置条件:
#   需要先运行 ./build.sh 构建包，或确保 dist/ 目录存在
#
# Token 配置 (PyPI 已不支持密码登录，必须使用 API Token):
#   方式 1: 环境变量 UV_PUBLISH_TOKEN
#   方式 2: keyring set https://upload.pypi.org/legacy/ __token__
#
# 注意: uv publish 会自动从 keyring 读取凭据，或使用环境变量中的 token

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认参数
TARGET="${1:-prod}"

# 打印函数
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查 uv
check_uv() {
    if ! command -v uv &> /dev/null; then
        error "未找到 uv，请先安装:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    success "uv $(uv --version | head -1)"
}

# 检查构建产物
check_dist() {
    info "检查构建产物..."
    
    if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
        error "dist/ 目录不存在或为空"
        echo ""
        warning "请先运行 ./build.sh 构建包"
        exit 1
    fi
    
    # 检查文件是否存在
    WHEEL_FILE=$(ls dist/*.whl 2>/dev/null | head -n 1)
    SDIST_FILE=$(ls dist/*.tar.gz 2>/dev/null | head -n 1)
    
    if [ -z "$WHEEL_FILE" ]; then
        error "未找到 wheel 文件 (.whl)"
        warning "请先运行 ./build.sh 构建包"
        exit 1
    fi
    
    if [ -z "$SDIST_FILE" ]; then
        error "未找到源码分发文件 (.tar.gz)"
        warning "请先运行 ./build.sh 构建包"
        exit 1
    fi
    
    info "找到构建产物:"
    echo "  - Wheel: $(basename "$WHEEL_FILE")"
    echo "  - Source: $(basename "$SDIST_FILE")"
}

# 配置 Token
setup_token() {
    if [ -z "$UV_PUBLISH_TOKEN" ]; then
        # 检查 keyring 中是否已配置
        if keyring get https://upload.pypi.org/legacy/ __token__ &>/dev/null; then
            info "检测到 keyring 中已配置 PyPI 凭据"
            success "将使用 keyring 中的凭据进行认证"
        else
            warning "未设置 UV_PUBLISH_TOKEN 环境变量"
            info ""
            info "Token 配置方式 (PyPI 必须使用 API Token):"
            echo "  1. 环境变量: export UV_PUBLISH_TOKEN='pypi-xxxx...'"
            echo "  2. keyring 配置: keyring set https://upload.pypi.org/legacy/ __token__"
            echo ""
            info "获取 Token: https://pypi.org/manage/account/token/"
            echo ""
            warning "当前没有配置任何认证方式，uv publish 将提示输入凭据"
        fi
    else
        success "将使用 UV_PUBLISH_TOKEN 环境变量进行认证"
    fi
}

# 发布
publish() {
    local pypi_name pypi_url
    
    if [ "$TARGET" = "test" ]; then
        pypi_name="测试 PyPI (test.pypi.org)"
        pypi_url="https://test.pypi.org/legacy/"
    else
        pypi_name="正式 PyPI (pypi.org)"
        pypi_url=""
    fi
    
    echo ""
    echo "=========================================="
    warning "即将发布到 $pypi_name"
    echo "=========================================="
    echo ""
    info "构建产物:"
    ls -lh dist/
    echo ""
    
    read -p "确认发布? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        info "已取消发布"
        exit 0
    fi
    
    info "开始上传..."
    
    # 构建 uv publish 命令
    if [ "$TARGET" = "test" ]; then
        # 测试 PyPI
        if [ -n "$UV_PUBLISH_TOKEN" ]; then
            uv publish --publish-url "$pypi_url" --token "$UV_PUBLISH_TOKEN"
        else
            # 没有 token，uv 会从 keyring 读取或提示输入
            uv publish --publish-url "$pypi_url"
        fi
    else
        # 正式 PyPI (默认 PyPI 地址)
        if [ -n "$UV_PUBLISH_TOKEN" ]; then
            uv publish --token "$UV_PUBLISH_TOKEN"
        else
            # 没有 token，uv 会从 keyring 读取或提示输入
            uv publish
        fi
    fi
    
    success "发布完成！"
}

# 显示帮助
show_help() {
    echo "AuriMyth Foundation Kit - PyPI 发布工具"
    echo ""
    echo "使用方法: ./publish.sh [test|prod]"
    echo ""
    echo "参数:"
    echo "  test    发布到测试 PyPI"
    echo "  prod    发布到正式 PyPI (默认)"
    echo ""
    echo "前置条件:"
    echo "  需要先运行 ./build.sh 构建包，或确保 dist/ 目录存在"
    echo ""
    echo "Token 配置 (PyPI 必须使用 API Token):"
    echo ""
    echo "  方式 1: 环境变量"
    echo "    export UV_PUBLISH_TOKEN='pypi-xxxx...'"
    echo "    ./publish.sh prod"
    echo ""
    echo "  方式 2: keyring (推荐)"
    echo "    keyring set https://upload.pypi.org/legacy/ __token__"
    echo "    然后输入你的 PyPI Token"
    echo "    ./publish.sh prod"
    echo ""
    echo "获取 Token: https://pypi.org/manage/account/token/"
}

# 主流程
main() {
    # 帮助信息
    if [ "$TARGET" = "-h" ] || [ "$TARGET" = "--help" ]; then
        show_help
        exit 0
    fi
    
    # 验证参数
    if [ "$TARGET" != "test" ] && [ "$TARGET" != "prod" ]; then
        error "无效参数: $TARGET"
        echo "使用 ./publish.sh --help 查看帮助"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "  AuriMyth Foundation Kit - PyPI 发布"
    echo "  使用 uv + hatch-vcs"
    echo "=========================================="
    echo ""
    
    if [ "$TARGET" = "test" ]; then
        info "目标: ${YELLOW}测试 PyPI${NC}"
    else
        info "目标: ${GREEN}正式 PyPI${NC}"
    fi
    echo ""
    
    check_uv
    echo ""
    
    check_dist
    echo ""
    
    setup_token
    echo ""
    
    publish
    
    echo ""
    success "发布流程完成！"
}

main
