import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true);
}

function addPreviewWidget(nodeType, nodeData, widgetName="audio", when="onNodeCreated") {
        
    switch (when){
        case "onNodeCreated":
            // TODO: not working
            // chainCallback(nodeType.prototype, "onNodeCreated", function () {
            //     const audioWidget = this.widgets.find((w) => w.name === widgetName);
            //     if (!audioWidget || !audioWidget?.value) return null
                
            //     const widgetId = `opt_widget-${Math.random()}`
            //     let options = {
            //         widgetId, type: "input", filename: audioWidget.value
            //     }
            //     nodeData.input.hidden = {
            //         ...nodeData.input.hidden,
            //         [widgetId]: ["AUDIOPREVIEW",options]
            //     };
            //     previewAudio(this,options)
            //     audioWidget.callback = function () {
            //         options = {
            //             widgetId, type: "input", filename: audioWidget.value
            //         }
            //         previewAudio(this,options)
            //         if (cb) {
            //             return cb.apply(this, arguments);
            //         }
            //     };
            // })
            break;
        case "onExecuted":
            chainCallback(nodeType.prototype, when, function (data) {
                const widgetId = `opt_widget-${Math.random()}`
                nodeData.input.hidden = {
                    ...nodeData.input.hidden,
                    [widgetId]: ["AUDIOPREVIEW",{widgetName,...data.preview[0]}]
                };
                previewAudio(this,data.preview[0])
            })
            break;

        default: console.log({nodeType, nodeData, widgetName, when, node: this})
    }
}

function previewAudio(node,options={filename: null, type: "input", widgetName: "audio", widgetId: null}){
    // while (node.widgets.length > 2){
    //     node.widgets.pop();
    // }
    try {
        console.log({node, options})
        var el = document.getElementById(options.widgetId);
        el?.remove();
    } catch (error) {
        console.log(error);
    }
    var element = document.createElement("div");
    element.id = options.widgetId
    const previewNode = node;
    var previewWidget = node.addDOMWidget("audiopreview", "preview", element, {
        serialize: false,
        hideOnZoom: false,
        getValue() {
            return element.value;
        },
        setValue(v) {
            element.value = v;
        },
    });
    previewWidget.computeSize = function(width) {
        if (node.aspectRatio && !node.parentEl.hidden) {
            let height = (previewNode.size[0]-20)/ node.aspectRatio + 10;
            if (!(height > 0)) {
                height = 0;
            }
            node.computedHeight = height + 10;
            return [width, height];
        }
        return [width, -4];//no loaded src, widget should not display
    }
    // element.style['pointer-events'] = "none"
    previewWidget.value = {hidden: false, paused: false, params: {}}
    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "audio_preview";
    previewWidget.parentEl.style['width'] = "100%"
    element.appendChild(previewWidget.parentEl);
    previewWidget.audioEl = document.createElement("audio");
    previewWidget.audioEl.controls = true;
    previewWidget.audioEl.loop = false;
    previewWidget.audioEl.muted = false;
    previewWidget.audioEl.style['width'] = "100%"
    previewWidget.audioEl.addEventListener("loadedmetadata", () => {
        previewWidget.aspectRatio = previewWidget.audioEl.audioWidth / previewWidget.audioEl.audioHeight;
        fitHeight(node);
    });
    previewWidget.audioEl.addEventListener("error", () => {
        //TODO: consider a way to properly notify the user why a preview isn't shown.
        previewWidget.parentEl.hidden = true;
        fitHeight(node);
    });

    let params =  options;
    
    previewWidget.parentEl.hidden = previewWidget.value.hidden;
    previewWidget.audioEl.autoplay = !previewWidget.value.paused && !previewWidget.value.hidden && Boolean(options.autoplay);
    let target_width = 256
    if (element.style?.width) {
        //overscale to allow scrolling. Endpoint won't return higher than native
        target_width = element.style.width.slice(0,-2)*2;
    }
    if (!params.force_size || params.force_size.includes("?") || params.force_size == "Disabled") {
        params.force_size = target_width+"x?"
    } else {
        let size = params.force_size.split("x")
        let ar = parseInt(size[0])/parseInt(size[1])
        params.force_size = target_width+"x"+(target_width/ar)
    }
    
    previewWidget.audioEl.src = api.apiURL('/view?' + new URLSearchParams(params));

    previewWidget.audioEl.hidden = false;
    previewWidget.parentEl.appendChild(previewWidget.audioEl)
    return previewWidget
}

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}

async function uploadFile(file) {
    //TODO: Add uploaded file to cache with Cache.put()?
    try {
        // Wrap file in formdata so it includes filename
        const body = new FormData();
        const i = file.webkitRelativePath.lastIndexOf('/');
        const subfolder = file.webkitRelativePath.slice(0,i+1)
        const new_file = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", new_file);
        if (i > 0) {
            body.append("subfolder", subfolder);
        }
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            return resp.status
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
}

function addUploadWidget(nodeType, nodeData, widgetName, type="video") {
    
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const fileInput = document.createElement("input");
        chainCallback(nodeType.prototype, "onRemoved", () => {
            fileInput?.remove();
        });
        if (type == "folder") {
            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                webkitdirectory: true,
                onchange: async () => {
                    const directory = fileInput.files[0].webkitRelativePath;
                    const i = directory.lastIndexOf('/');
                    if (i <= 0) {
                        throw "No directory found";
                    }
                    const path = directory.slice(0,directory.lastIndexOf('/'))
                    if (pathWidget.options.values.includes(path)) {
                        alert("A folder of the same name already exists");
                        return;
                    }
                    let successes = 0;
                    for(const file of fileInput.files) {
                        if (await uploadFile(file) == 200) {
                            successes++;
                        } else {
                            //Upload failed, but some prior uploads may have succeeded
                            //Stop future uploads to prevent cascading failures
                            //and only add to list if an upload has succeeded
                            if (successes > 0) {
                                break
                            } else {
                                return;
                            }
                        }
                    }
                    pathWidget.options.values.push(path);
                    pathWidget.value = path;
                    if (pathWidget.callback) {
                        pathWidget.callback(path)
                    }
                },
            });
        } else if (type == "video") {
            Object.assign(fileInput, {
                type: "file",
                accept: "video/webm,video/mp4,video/mkv,image/gif",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        if (await uploadFile(fileInput.files[0]) != 200) {
                            //upload failed and file can not be added to options
                            return;
                        }
                        const filename = fileInput.files[0].name;
                        pathWidget.options.values.push(filename);
                        pathWidget.value = filename;
                        if (pathWidget.callback) {
                            pathWidget.callback(filename)
                        }
                    }
                },
            });
        } else if (type == "audio") {
            Object.assign(fileInput, {
                type: "file",
                accept: "audio/mpeg,audio/wav,audio/x-wav,audio/ogg,audio/flac",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        if (await uploadFile(fileInput.files[0]) != 200) {
                            //upload failed and file can not be added to options
                            return;
                        }
                        const filename = fileInput.files[0].name;
                        pathWidget.options.values.push(filename);
                        pathWidget.value = filename;
                        if (pathWidget.callback) {
                            pathWidget.callback(filename)
                        }
                        previewAudio(this, {filename, type: "input", widgetId: filename})
                    }
                },
            });
        }else {
            throw "Unknown upload type"
        }
        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
            //clear the active click event
            app.canvas.node_widget = null

            fileInput.click();
        });
        uploadWidget.options.serialize = false;
    });
}

app.registerExtension({
	name: "RVC-Studio.UI",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.category?.includes("RVC")){
            switch (nodeData?.name){
                case "RVC-Studio.LoadAudio": 
                    addUploadWidget(nodeType, nodeData, "audio", "audio")
                    addPreviewWidget(nodeType, nodeData, "audio", "onNodeCreated" )
                    break
                case "DownloadAudio":
                    addPreviewWidget(nodeType, nodeData, "audio", "onExecuted" )
                    break
                case "RVC-Studio.PreviewAudio":
                    addPreviewWidget(nodeType, nodeData, "audio", "onExecuted" )
                    break
                case "MergeAudioNode":
                    addPreviewWidget(nodeType, nodeData, "audio", "onExecuted" )
                    break;
                case "RVCNode":
                    addPreviewWidget(nodeType, nodeData, "audio", "onExecuted" )
                    break;

                case "SliceNode":
                    chainCallback(nodeType.prototype, "onConnectInput", function (_, inputs) {
                        this.outputs[0].name = inputs;
                        this.outputs[0].type = inputs;
                    })
                    break;

                default:
                    
                    break
            }
        }
	},
});

